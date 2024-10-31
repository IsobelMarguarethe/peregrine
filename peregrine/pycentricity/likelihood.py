import bilby as bb
import numpy as np
import pycentricity.overlap as ovlp
import pycentricity.waveform as wf
import numpy.random as random

from scipy.interpolate import interp1d
from scipy.special import i0e, logsumexp

from collections import namedtuple


_CalculatedSNRs = namedtuple('CalculatedSNRs',
                                 ['d_inner_h',
                                  'optimal_snr_squared',
                                  'complex_matched_filter_snr',
                                  'd_inner_h_squared_tc_array'])

def phase_marginalized_likelihood(d_inner_h, h_inner_h):
    # IRS TODO write docstring
    _bessel_function_interped = interp1d(
            np.logspace(-5, 10, int(1e6)), np.logspace(-5, 10, int(1e6)) +
            np.log([i0e(snr) for snr in np.logspace(-5, 10, int(1e6))]),
            bounds_error=False, fill_value=(0, np.nan))
    d_inner_h = _bessel_function_interped(abs(d_inner_h))
    return d_inner_h - h_inner_h / 2


def time_marginalized_likelihood(d_inner_h_tc_array, h_inner_h, priors, sampling_frequency, interferometers):
    # IRS TODO write docstring
    log_l_tc_array = phase_marginalized_likelihood(
                d_inner_h=d_inner_h_tc_array,
                h_inner_h=h_inner_h)
    #print('phase marginalised likelihood: {}'.format(log_l_tc_array))
    delta_tc = 2 / sampling_frequency
    times = interferometers.start_time + np.linspace(
                0, interferometers.duration,
                int(interferometers.duration / 2 *
                    sampling_frequency + 1))[1:]
    time_prior_array = priors['geocent_time'].prob(times) * delta_tc
    return logsumexp(log_l_tc_array, b=time_prior_array)


def calculate_snrs(waveform_polarizations, interferometer, parameters, duration):
    # IRS TODO add docstring
    signal = interferometer.get_detector_response(waveform_polarizations, parameters)
    d_inner_h = interferometer.inner_product(signal=signal)
    optimal_snr_squared = interferometer.optimal_snr_squared(signal=signal)
    complex_matched_filter_snr = d_inner_h / (optimal_snr_squared**0.5)

    d_inner_h_squared_tc_array =\
                4 / duration * np.fft.fft(
                    signal[0:-1] *
                    interferometer.frequency_domain_strain.conjugate()[0:-1] /
                    interferometer.power_spectral_density_array[0:-1])
    return _CalculatedSNRs(
            d_inner_h=d_inner_h, optimal_snr_squared=optimal_snr_squared,
            complex_matched_filter_snr=complex_matched_filter_snr,
            d_inner_h_squared_tc_array=d_inner_h_squared_tc_array)

   

def phase_time_marginalized_log_likelihood_ratio(waveform_polarizations, interferometers, 
                                                 parameters, duration, priors, sampling_frequency):
    # IRS TODO add docstring
    #print("calculating phase-time marginalised log-likelihood ratio") 
    d_inner_h = 0.
    optimal_snr_squared = 0.
    complex_matched_filter_snr = 0.
    d_inner_h_tc_array = np.zeros(interferometers.frequency_array[0:-1].shape, dtype=np.complex128)

    for interferometer in interferometers:
        per_detector_snr = calculate_snrs(
                waveform_polarizations=waveform_polarizations,
                interferometer=interferometer,
                parameters=parameters,
                duration=duration)
        #print('SNR for {}: {}'.format(interferometer.name, per_detector_snr))

        d_inner_h += per_detector_snr.d_inner_h
        optimal_snr_squared += np.real(per_detector_snr.optimal_snr_squared)
        complex_matched_filter_snr += per_detector_snr.complex_matched_filter_snr

        d_inner_h_tc_array += per_detector_snr.d_inner_h_squared_tc_array
    log_l = time_marginalized_likelihood(
                d_inner_h_tc_array=d_inner_h_tc_array,
                h_inner_h=optimal_snr_squared,
                priors=priors,
                sampling_frequency=sampling_frequency, 
                interferometers=interferometers)
    #print('time and phase marginalised likelihood: {}'.format(log_l))
    #print('real part: {}'.format(float(log_l.real)))
    return float(log_l.real)


def log_likelihood_interferometer(
    waveform_polarizations, interferometer, parameters, duration
):
    """
    Return the log likelihood at a single interferometer.
    :param waveform_polarizations: dict
        frequency-domain waveform polarisations
    :param interferometer: Interferometer
        Interferometer object
    :param parameters: dict
        waveform parameters
    :param duration: int
        time duration of the signal
    :return:
        log_l: float
            log likelihood evaluation
    """
    signal_ifo = interferometer.get_detector_response(
        waveform_polarizations, parameters
    )
    log_l = (
        -2.0
        / duration
        * np.vdot(
            interferometer.frequency_domain_strain - signal_ifo,
            (interferometer.frequency_domain_strain - signal_ifo)
            / interferometer.power_spectral_density_array,
        )
    )
    return log_l.real


def noise_log_likelihood(interferometers, duration):
    """
    Return the likelihood that the data is caused by noise
    :param interferometers: InterferometerList
        list of interferometers involved in the detection
    :param duration: int
        time duration of the signal
    :return:
        log_l: float
        log likelihood evaluation
    """
    log_l = 0
    for interferometer in interferometers:
        log_l -= (
            2.0
            / duration
            * np.sum(
                abs(interferometer.frequency_domain_strain) ** 2
                / interferometer.power_spectral_density_array
            )
        )
    return log_l.real


def log_likelihood(waveform_polarizations, interferometers, parameters, duration):
    """
    Return the log likelihood of all interferometers involved in the detection
    :param waveform_polarizations: dict
        frequency-domain waveform polarisations
    :param interferometers: InterferometerList
        list of interferometers involved in the detection
    :param parameters: dict
        waveform parameters
    :param duration: int
        time duration of the signal
    :return:
        log_l: float
        log likelihood evaluation
    """
    log_l = 0
    for interferometer in interferometers:
        log_l += log_likelihood_interferometer(
            waveform_polarizations, interferometer, parameters, duration
        )
    return log_l.real


def log_likelihood_ratio(waveform_polarizations, interferometers, parameters, duration):
    """
    Return the ratio of the likelihood of the signal being real to the likelihood
    of the signal being noise
    :param waveform_polarizations: dict
        frequency-domain waveform polarisations
    :param interferometers: InterferometerList
        list of interferometers involved in the relationship
    :param parameters: dict
        waveform variables
    :param duration: int
        time duration of the signal
    :return:
        log_likelihood_ratio: float
            log likelihood evaluation ratio
    """
    return log_likelihood(
        waveform_polarizations, interferometers, parameters, duration
    ) - noise_log_likelihood(interferometers, duration)


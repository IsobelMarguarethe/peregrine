"""

A file for generating waveforms and related objects for the pycentricity package.

"""
from copy import deepcopy
import bilby as bb
import numpy as np
import os

from scipy import signal
from pycentricity.pyWaveformGenerator.waveform import calculate_waveform, calculate_waveform_ep


GMsun_c3 = 6.67384e-11 * 1.988409870698050731911960804878414216e30 / (299792458e0)**3

def f_kep2peri(f_kep, e):

    num = np.sqrt(1-e**2)
    denom = (1-e)**2
    return f_kep*num/denom


def generate_seobnrpe_waveform(parameters, info):
    """
    Return the waveform polarisations simulated by the SEOBNRPE python code.
    :param parameters: dict
        dictionary of waveform parameters
            x(m1, m2, s1x, s1y, s1z, s2x, s2y, s2z, e0, dL, zeta_rad, iota_rad, beta_rad, Phi_rad)
            > m1, m2: component mass of BH in solar mass,
            > s1x, s1y, s1z, s2x, s2y, s2z: dimentionless spin vector chi of BH, the norm of these 
              vectors should less than 1
            > e0 initial eccentricity at given reference frequency Mf_ref
            > dL luminosity distance in Mpc
            > zeta_rad relativistic anomaly zeta in r = p / (1 + e cos(zeta)).
            > iota_rad inclination angle in rad
            > beta_rad represent the initial direction of major axis of the elliptical orbit,
                which is equivalent to the phiRef parameter in lalsim-inspiral, HALF of main GW phase at reference
                so the input phase needs to be halved
            > Phi_rad the complex angle of h+ - ihx at the merge stage (default unused other
                than you set use_coaphase=True). Think equivalent to psi in lal but not sure.
                Possibly also needs to be halved? Or doubled? But is not used until the antenna pattern function.
    :param info: dict
        dictionary of extra parameters
    :return:
        hp, hc: numpy arrays
            time-domain plus and cross waveform polarisations
    """
    # Conversions to parameters we need to pass to the waveform function
    f_peri = f_kep2peri(info['reference_frequency'], parameters['eccentricity'])
    Mf_ref = GMsun_c3 * (parameters['mass_1'] + parameters['mass_2']) * f_peri
    beta_rad = parameters['phase'] #- np.pi / 2
    Phi_rad = parameters['psi']
    # Conversion of spin parameters to component spins
    iota, spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z = bb.gw.conversion.bilby_to_lalsimulation_spins(
                                                parameters['theta_jn'],
                                                parameters['phi_jl'], parameters['tilt_1'],
                                                parameters['tilt_2'], parameters['phi_12'], parameters['a_1'],
                                                parameters['a_2'], parameters['mass_1'], parameters['mass_2'],
                                                info['reference_frequency'], parameters['phase'])

    waveform, dynamics = calculate_waveform_ep((parameters['mass_1'], parameters['mass_2'],
                                                spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
                                                parameters['eccentricity'], parameters['luminosity_distance'],
                                                parameters['relativistic_anomaly'],
                                                iota, beta_rad, Phi_rad), # Not sure if there should be a difference here
                                                f_min=info['minimum_frequency'],
                                                Mf_ref=Mf_ref,
                                                is_coframe=False, # Have tested this - unlikely to be True
                                                srate=info['sampling_frequency'],
                                                use_coaphase=False, # Not sure if this should be true or false, but think False because overlap with PhenomXPMH does not change when psi is changed (which implies use_coaphase=False in XPHM as well 
                                                is_only22=info['is_only22'])
    try:
        wf = -waveform.hpc.value
        hp = wf.real
        hc = -wf.imag
    except AttributeError:
        print('Bad Parameters: {}'.format(parameters))
        hp = [0]
        hc = [0]

    return hp, hc


def apply_window(waveform):
    window = signal.windows.exponential(int(len(waveform['plus'])/4),
                                        None, len(waveform['plus'])/100)
    # Custom window - set to 1 from where window = 1, avoid cutting off the merger and ringdown
    mid_index = int(len(window)/2)
    window[mid_index::] = 1
    pad_width = len(waveform['plus']) - len(window)
    window = np.pad(window, (0, pad_width), 'constant', constant_values=(0,1))
    waveform = {key: waveform[key] * window for key in waveform}
    return window, waveform


def seobnrpe_bbh_with_precessing_spin_and_eccentricity(
    parameters, sampling_frequency, minimum_frequency, reference_frequency, is_only22=False
):
    """
    Return the  waveform polarisations.
    :param parameters: dict
        dictionary of waveform parameters
    :param sampling_frequency: int
        frequency with which to 'sample' the waveform
    :param minimum_frequency: int
        minimum frequency to contain in the signal
    :return:
        seobnre: dict
            time-domain waveform polarisations
    """
    info = {'minimum_frequency': minimum_frequency, 'reference_frequency': reference_frequency,
            'sampling_frequency': sampling_frequency, 'is_only22': is_only22}
    hp, hc = generate_seobnrpe_waveform(parameters, info)
    seobnre_waveform = {
        "plus": hp,
        "cross": hc,
    }
    return seobnre_waveform


def seobnrpe_bbh_waveform_model(
        time_array, mass_1, mass_2, a_1, a_2, phi_12, phi_jl, tilt_1, tilt_2, phase, luminosity_distance,
        theta_jn, eccentricity, relativistic_anomaly, geocent_time, psi, **waveform_kwargs):
    parameters = {'mass_1': mass_1, 'mass_2': mass_2, 'lambda_1': 0, 'lambda_2': 0, 'a_1': a_1,
                  'a_2': a_2, 'phi_12': phi_12, 'phi_jl': phi_jl, 'tilt_1': tilt_1, 'tilt_2': tilt_2,
                  'phase': phase, 'psi': psi, 'luminosity_distance': luminosity_distance,
                  'theta_jn': theta_jn, 'eccentricity': eccentricity,
                  'relativistic_anomaly': relativistic_anomaly}
    waveform = seobnrpe_bbh_with_precessing_spin_and_eccentricity(parameters, waveform_kwargs['sampling_frequency'],
                                              waveform_kwargs['minimum_frequency'], waveform_kwargs['reference_frequency'],
                                              is_only22=waveform_kwargs['is_only22'])
    if waveform['plus'] is None:
        return waveform
    else:
        # Make sure the signal is the right length
        time_length = len(time_array)
        length_difference = len(waveform['plus']) - time_length

        if length_difference < 0:
            # Waveform is too short; needs zero-padding, but window applied first
            window, waveform = apply_window(waveform)
            temp_wf = {key: np.pad(waveform[key], (np.abs(length_difference), 0),
                             'constant', constant_values=(0, 0)) for key in waveform}
            waveform = temp_wf
        elif length_difference > 0:
            # Waveform is too long; needs cutting down to size, then windowing
            temp_wf = {key: waveform[key][length_difference:] for key in waveform}
            window, waveform = apply_window(temp_wf)
        else:
            pass
        # Wrap to bring the coalescence to the end of the time series
        coalescence_index = int(np.argmax(abs(waveform['plus'] - 1j * waveform['cross'])))
        roll_shift = time_length - coalescence_index - 1
        temp_wf = {key: np.roll(waveform[key], roll_shift) for key in waveform}
        seob_bbh_waveform = temp_wf
        # Return the waveform
        return seob_bbh_waveform


def get_seobnrpe_waveform_generator(
    minimum_frequency, reference_frequency, sampling_frequency, duration, start_time,
    parameter_conversion=None, is_only22=False
):
    waveform_arguments = dict(
        sampling_frequency=sampling_frequency,
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency,
        duration=duration,
        is_only22=is_only22
    )
    waveform_generator = bb.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        time_domain_source_model=seobnrpe_bbh_waveform_model,
        waveform_arguments=waveform_arguments,
        start_time=start_time,
        parameter_conversion=parameter_conversion
    )
    return waveform_generator


def get_comparison_waveform_generator(
    minimum_frequency, sampling_frequency, duration,
        maximum_frequency, reference_frequency, waveform_approximant="SEOBNRv4P",
         parameter_conversion=None
):
    """
    Provide the waveform generator object for the comparison waveform,
    which can be generically chosen.
    :param minimum_frequency: int
        minimum frequency to contain in the waveform
    :param sampling_frequency: int
        frequency with which to 'sample' the signal
    :param duration: int
        time duration of the signal
    :return:
        waveform_generator: WaveformGenerator
            the waveform generator object for the comparison waveform
    """
    waveform_arguments = dict(
        waveform_approximant=waveform_approximant,
        reference_frequency=reference_frequency,
        minimum_frequency=minimum_frequency,
        maximum_frequency=maximum_frequency,
    )
    waveform_generator = bb.gw.WaveformGenerator(
        duration=duration,
        sampling_frequency=sampling_frequency,
        frequency_domain_source_model=bb.gw.source.lal_binary_black_hole,
        waveform_arguments=waveform_arguments,
        parameter_conversion=parameter_conversion
    )
    return waveform_generator

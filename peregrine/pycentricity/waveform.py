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

class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration

    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False

def SpinWeightedM2SphericalHarmonic(theta, phi, l, m):
    modetag = l*100 + m
    GET_SQRT = np.sqrt
    CST_PI = np.pi
    GET_COS = np.cos
    GET_SIN = np.sin
    GET_POW = np.power
    for case in switch(modetag):
        if case(198):
            # mode 2, -2
            fac = GET_SQRT( 5.0 / ( 64.0 * CST_PI ) ) * ( 1.0 - GET_COS( theta ))*( 1.0 - GET_COS( theta ))
            break
        if case(199):
            # mode 2, -1
            fac = GET_SQRT( 5.0 / ( 16.0 * CST_PI ) ) * GET_SIN( theta )*( 1.0 - GET_COS( theta ))
            break
        if case(200):
            # mode 2, 0
            fac = GET_SQRT( 15.0 / ( 32.0 * CST_PI ) ) * GET_SIN( theta )*GET_SIN( theta )
            break
        if case(201):
            # mode 2, 1
            fac = GET_SQRT( 5.0 / ( 16.0 * CST_PI ) ) * GET_SIN( theta )*( 1.0 + GET_COS( theta ))
            break
        if case(202):
            # mode 2, 2
            fac = GET_SQRT( 5.0 / ( 64.0 * CST_PI ) ) * ( 1.0 + GET_COS( theta ))*( 1.0 + GET_COS( theta ))
            break
        if case(297):
            # mode 3, -3
            fac = GET_SQRT(21.0/(2.0*CST_PI))*GET_COS(theta/2.0)*GET_POW(GET_SIN(theta/2.0),5.0)
            break
        if case(298):
            # mode 3, -2
            fac = GET_SQRT(7.0/(4.0*CST_PI))*(2.0 + 3.0*GET_COS(theta))*GET_POW(GET_SIN(theta/2.0),4.0)
            break
        if case(299):
            # mode 3, -1
            fac = GET_SQRT(35.0/(2.0*CST_PI))*(GET_SIN(theta) + 4.0*GET_SIN(2.0*theta) - 3.0*GET_SIN(3.0*theta))/32.0
            break
        if case(300):
            # mode 3, 0
            fac = (GET_SQRT(105.0/(2.0*CST_PI))*GET_COS(theta)*GET_POW(GET_SIN(theta),2.0))/4.0
            break
        if case(301):
            # mode 3, 1
            fac = -GET_SQRT(35.0/(2.0*CST_PI))*(GET_SIN(theta) - 4.0*GET_SIN(2.0*theta) - 3.0*GET_SIN(3.0*theta))/32.0
            break
        if case(302):
            # mode 3, 2
            fac = GET_SQRT(7.0/CST_PI)*GET_POW(GET_COS(theta/2.0),4.0)*(-2.0 + 3.0*GET_COS(theta))/2.0
            break
        if case(303):
            # mode 3, 3
            fac = -GET_SQRT(21.0/(2.0*CST_PI))*GET_POW(GET_COS(theta/2.0),5.0)*GET_SIN(theta/2.0)
            break
        if case(396):
            # mode 4, -4
            fac = 3.0*GET_SQRT(7.0/CST_PI)*GET_POW(GET_COS(theta/2.0),2.0)*GET_POW(GET_SIN(theta/2.0),6.0)
            break
        if case(397):
            # mode 4, -3
            fac = 3.0*GET_SQRT(7.0/(2.0*CST_PI))*GET_COS(theta/2.0)*(1.0 + 2.0*GET_COS(theta))*GET_POW(GET_SIN(theta/2.0),5.0)
            break
        if case(398):
            # mode 4, -2
            fac = (3.0*(9.0 + 14.0*GET_COS(theta) + 7.0*GET_COS(2.0*theta))*GET_POW(GET_SIN(theta/2.0),4.0))/(4.0*GET_SQRT(CST_PI))
            break
        if case(399):
            # mode 4, -1
            fac = (3.0*(3.0*GET_SIN(theta) + 2.0*GET_SIN(2.0*theta) + 7.0*GET_SIN(3.0*theta) - 7.0*GET_SIN(4.0*theta)))/(32.0*GET_SQRT(2.0*CST_PI))
            break
        if case(400):
            # mode 4, 0
            fac = (3.0*GET_SQRT(5.0/(2.0*CST_PI))*(5.0 + 7.0*GET_COS(2.0*theta))*GET_POW(GET_SIN(theta),2.0))/16.0
            break
        if case(401):
            # mode 4, 1
            fac = (3.0*(3.0*GET_SIN(theta) - 2.0*GET_SIN(2.0*theta) + 7.0*GET_SIN(3.0*theta) + 7.0*GET_SIN(4.0*theta)))/(32.0*GET_SQRT(2.0*CST_PI))
            break
        if case(402):
            # mode 4, 2
            fac = (3.0*GET_POW(GET_COS(theta/2.0),4.0)*(9.0 - 14.0*GET_COS(theta) + 7.0*GET_COS(2.0*theta)))/(4.0*GET_SQRT(CST_PI))
            break
        if case(403):
            # mode 4, 3
            fac = -3.0*GET_SQRT(7.0/(2.0*CST_PI))*GET_POW(GET_COS(theta/2.0),5.0)*(-1.0 + 2.0*GET_COS(theta))*GET_SIN(theta/2.0)
            break
        if case(404):
            # mode 4, 4
            fac = 3.0*GET_SQRT(7.0/CST_PI)*GET_POW(GET_COS(theta/2.0),6.0)*GET_POW(GET_SIN(theta/2.0),2.0)
            break
        if case(495):
            fac = GET_SQRT(330.0/CST_PI)*GET_POW(GET_COS(theta/2.0),3.0)*GET_POW(GET_SIN(theta/2.0),7.0)
            break
        if case(496):
            fac = GET_SQRT(33.0/CST_PI)*GET_POW(GET_COS(theta/2.0),2.0)*(2.0 + 5.0*GET_COS(theta))*GET_POW(GET_SIN(theta/2.0),6.0)
            break
        if case(497):
            fac = (GET_SQRT(33.0/(2.0*CST_PI))*GET_COS(theta/2.0)*(17.0 + 24.0*GET_COS(theta) + 15.0*GET_COS(2.0*theta))*GET_POW(GET_SIN(theta/2.0),5.0))/4.0
            break
        if case(498):
            fac = (GET_SQRT(11.0/CST_PI)*(32.0 + 57.0*GET_COS(theta) + 36.0*GET_COS(2.0*theta) + 15.0*GET_COS(3.0*theta))*GET_POW(GET_SIN(theta/2.0),4.0))/8.0
            break
        if case(499):
            fac = (GET_SQRT(77.0/CST_PI)*(2.0*GET_SIN(theta) + 8.0*GET_SIN(2.0*theta) + 3.0*GET_SIN(3.0*theta) + 12.0*GET_SIN(4.0*theta) - 15.0*GET_SIN(5.0*theta)))/256.0
            break
        if case(500):
            fac = (GET_SQRT(1155.0/(2.0*CST_PI))*(5.0*GET_COS(theta) + 3.0*GET_COS(3.0*theta))*GET_POW(GET_SIN(theta),2.0))/32.0
            break
        if case(501):
            fac = GET_SQRT(77.0/CST_PI)*(-2.0*GET_SIN(theta) + 8.0*GET_SIN(2.0*theta) - 3.0*GET_SIN(3.0*theta) + 12.0*GET_SIN(4.0*theta) + 15.0*GET_SIN(5.0*theta))/256.0
            break
        if case(502):
            fac = GET_SQRT(11.0/CST_PI)*GET_POW(GET_COS(theta/2.0),4.0)*(-32.0 + 57.0*GET_COS(theta) - 36.0*GET_COS(2.0*theta) + 15.0*GET_COS(3.0*theta))/8.0
            break
        if case(503):
            fac = -GET_SQRT(33.0/(2.0*CST_PI))*GET_POW(GET_COS(theta/2.0),5.0)*(17.0 - 24.0*GET_COS(theta) + 15.0*GET_COS(2.0*theta))*GET_SIN(theta/2.0)/4.0
            break
        if case(504):
            fac = GET_SQRT(33.0/CST_PI)*GET_POW(GET_COS(theta/2.0),6.0)*(-2.0 + 5.0*GET_COS(theta))*GET_POW(GET_SIN(theta/2.0),2.0)
            break
        if case(505):
            fac = -GET_SQRT(330.0/CST_PI)*GET_POW(GET_COS(theta/2.0),7.0)*GET_POW(GET_SIN(theta/2.0),3.0)
            break
        else:
            raise Exception(f'Unsupported mode {(l, m)}')
    return fac * np.exp(1.j* m * phi)


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
    Mf_ref = GMsun_c3 * (parameters['mass_1'] + parameters['mass_2']) * info['reference_frequency']
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

    '''
    # Calculate the scaling for the amplitude
    Msun_to_kg = 1.999999999e30
    Mpc_to_m = 3.086e22
    c = 299792458
    G = 6.67430e-11
    # IRSA really not sure about these factors. Need to talk to Xiaolin about them again.
    i = iota
    b = parameters['phase']
    factor = (parameters['mass_1'] + parameters['mass_2']) * Msun_to_kg * G /\
             (parameters['luminosity_distance'] * Mpc_to_m * c**2)
    '''
    try:
        wf = -waveform.hpc.value # np.array(waveform.h22) * SpinWeightedM2SphericalHarmonic(i, np.pi/2 - b, 2, 2) * factor
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

        #import matplotlib.pyplot as plt
        #fig, ax = plt.subplots(3,1, figsize=(8,10))
        if length_difference < 0:
            # Waveform is too short; needs zero-padding, but window applied first
            window, waveform = apply_window(waveform)
            temp_wf = {key: np.pad(waveform[key], (np.abs(length_difference), 0),
                             'constant', constant_values=(0, 0)) for key in waveform}
            waveform = temp_wf
            #ax[0].plot(waveform['plus']*1e22)
            #ax[0].plot(window*4)
            #ax[1].plot(waveform['cross']*1e22)
            #ax[1].plot(window*4)
        elif length_difference > 0:
            # Waveform is too long; needs cutting down to size, then windowing
            temp_wf = {key: waveform[key][length_difference:] for key in waveform}
            window, waveform = apply_window(temp_wf)
            #ax[0].plot(waveform['plus']*1e22)
            #ax[0].plot(window*4)
            #ax[1].plot(waveform['cross']*1e22)
            #ax[1].plot(window*4)
        else:
            pass
        # Wrap to bring the coalescence to the end of the time series
        coalescence_index = int(np.argmax(abs(waveform['plus'] - 1j * waveform['cross'])))
        roll_shift = time_length - coalescence_index - 1
        temp_wf = {key: np.roll(waveform[key], roll_shift) for key in waveform}
        seob_bbh_waveform = temp_wf
        #ax[2].plot(seob_bbh_waveform['plus']*1e22)
        #ax[2].plot(seob_bbh_waveform['cross']*1e22)
        #plt.savefig('debug_wf.png')
        #plt.close()
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

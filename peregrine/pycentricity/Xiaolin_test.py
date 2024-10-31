import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform
import sys, os
from pathlib import Path
import time
from pyWaveformGenerator.psd import GWDetector, DetectorPSD
from pyWaveformGenerator.waveform import calculate_waveform_ep, calculate_waveform
from pyWaveformGenerator import SEOBNRWaveformCaller, test_interface

def get_lalh(m1, m2, fmin, **kwargs):
    hp, hc = get_td_waveform(approximant = 'SEOBNRv4PHM',
                             mass1 = m1,
                             mass2 = m2,
                             f_lower = fmin,
                             delta_t = 1./16384,
                             **kwargs)
    return hp.sample_times, hp.data, hc.data

fmin = 10
m1 = 30
m2 = 25
s1x = 0.3
dL = 100
iota = np.pi/2
GMsun_c3 = 6.67384e-11 * 1.988409870698050731911960804878414216e30 / (299792458e0)**3
Mf_ref = GMsun_c3 * (m1 + m2) * fmin

t, hp, hc = get_lalh(m1, m2, fmin, distance=dL, inclination = iota, spin1x = s1x)
'''
calculate_waveform_ep((parameters['mass_1'], parameters['mass_2'],
                       spin_1x, spin_1y, spin_1z, spin_2x, spin_2y, spin_2z,
                       parameters['eccentricity'], parameters['luminosity_distance'],
                       parameters['relativistic_anomaly'],
                       iota, beta_rad, Phi_rad),
                       f_min=info['minimum_frequency'],
                       Mf_ref=Mf_ref, is_coframe=False, # Have tested this - unlikely to be True
                       srate=info['sampling_frequency'],
                       use_coaphase=True)
'''
waveform, dyn = calculate_waveform_ep((m1, m2, s1x, 0, 0, 0, 0, 0, 0, dL, 0, iota, 0, 0), fmin, Mf_ref=Mf_ref, 
                                       use_coaphase=False)

hpc_pycbc = hp - 1.j*hc
hpc_pw = -waveform.hpc.value

fig = plt.figure(figsize = (12, 6))
axs = fig.subplots(2, 1)
ax = axs[0]
ax.plot(t - t[0], np.real(hpc_pycbc))
ax.plot(waveform.time, np.real(hpc_pw))
ax = axs[1]
ax.plot(t - t[0], np.imag(hpc_pycbc))
ax.plot(waveform.time, np.imag(hpc_pw))
plt.savefig('Xiaolin_test.png')
plt.close()

import numpy as np
import matplotlib.pyplot as plt
from pycbc.waveform import get_td_waveform
import sys, os
from pathlib import Path
import time
from pyWaveformGenerator.psd import GWDetector, DetectorPSD
from pyWaveformGenerator.waveform import calculate_waveform_ep, calculate_waveform
from pyWaveformGenerator import SEOBNRWaveformCaller, test_interface

def get_lalh(m1, m2, fmin, approx = None, **kwargs):
    approx = 'IMRPhenomXPHM' if approx is None else approx
    hp, hc = get_td_waveform(approximant = approx,
                             mass1 = m1,
                             mass2 = m2,
                             f_lower = fmin,
                             f_ref = fmin,
                             delta_t = 1./16384,
                             **kwargs)
    return hp.sample_times, hp.data, hc.data

fmin = 20
m1 = 30
m2 = 25
t, hp, hc = get_lalh(m1, m2, fmin, distance=100, 
                     inclination = np.pi/2, spin1x = 0.3,
                     approx = 'SEOBNRv4PHM')
t2, hp2, hc2 = get_lalh(m1, m2, fmin, distance=100, 
                        inclination = np.pi/2, spin1x = 0.3,
                        approx = 'IMRPhenomXPHM')

mTScale = 4.925490947641266978197229498498379006e-6
Mf_ref = fmin * ((m1 + m2)*mTScale)
waveform_ep, dyn_ep = calculate_waveform_ep((m1, m2, 0.3, 0, 0, 0, 0, 0, 0, 100, 0, np.pi/2, 0, 0), fmin,
                                             Mf_ref=Mf_ref)

mTScale = 4.925490947641266978197229498498379006e-6

hpc_pycbc = hp - 1.j*hc
hpc_pycbc2 = hp2 - 1.j*hc2

hpc_pw_ep = -waveform_ep.hpc.value
t_ep = waveform_ep.time

ipeak = np.argmax(np.abs(hpc_pycbc))
ipeak2 = np.argmax(np.abs(hpc_pycbc2))
ipeak_ep = np.argmax(np.abs(hpc_pw_ep))

fig = plt.figure(figsize = (12, 6))
axs = fig.subplots(2, 1)
ax = axs[0]
ax.plot(t - t[ipeak], np.real(hpc_pycbc), color = 'black', label = 'SEOBNRv4PHM')
ax.plot(t2 - t2[ipeak2], np.real(hpc_pycbc2), color = 'blue', linestyle = '--', label = 'IMRPhenomXPHM')
ax.plot(t_ep - t_ep[ipeak_ep], np.real(hpc_pw_ep), color = 'red', linestyle = '-.', label = 'ep')

ax.set_xlim(-1, 0.1)
ax.legend()

ax = axs[1]
ax.plot(t - t[ipeak], np.imag(hpc_pycbc), color = 'black', label = 'SEOBNRv4PHM')
ax.plot(t2 - t2[ipeak2], np.imag(hpc_pycbc2), color = 'blue', linestyle = '--', label = 'IMRPhenomXPHM')
ax.plot(t_ep - t_ep[ipeak_ep], np.imag(hpc_pw_ep), color = 'red', linestyle = '-.', label = 'ep')

ax.set_xlim(-1, 0.1)

ax.legend()
plt.savefig('multi_comparison.png')
plt.close()



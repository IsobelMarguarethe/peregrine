print(
    r"""
             /'{>           Initialising PEREGRINE
         ____) (____        ----------------------
       //'--;   ;--'\\      Type: Load Simulator
      ///////\_/\\\\\\\     Authors: U.Bhardwaj, J.Alvey
             m m            Version: v0.0.1 | April 2023
"""
)

import sys
import numpy as np
import swyft.lightning as sl
from datetime import datetime
from config_utils import read_config, init_config
from simulator_utils import init_simulator

import pycentricity.waveform as wf
from bilby.gw import detector


import matplotlib.pyplot as plt

if __name__ == "__main__":
    args = sys.argv[1:]
    print(
        "{datetime.now().strftime('%a %d %b %H:%M:%S')} | [load_simulator.py] | Reading config file"
    )
    # Load and parse config file
    tmnre_parser = read_config(args)
    conf = init_config(tmnre_parser, args, sim=True)
    print(conf['waveform_params'])

    simulator = init_simulator(conf)

    sample = simulator.sample()
    int_vals = sample['z_int']
    params = {}
    for i, key in enumerate(simulator.int_priors.keys()):
        print(key, '=', int_vals[i])
        params[key] = int_vals[i]
    ext_vals = sample['z_ext']
    for i, key in enumerate(simulator.ext_priors.keys()):
        print(key, '=', ext_vals[i])
        params[key] = ext_vals[i]

    with open('sample.txt', 'w') as f:
        for key in sample.keys():
            f.write('{}: {}\n'.format(key, sample[key]))

    # generate a comparison seobnrpe waveform from scratch
    wf_gen = wf.get_seobnrpe_waveform_generator(conf['waveform_params']['minimum_frequency'], conf['waveform_params']['reference_frequency'],
                                                conf['waveform_params']['sampling_frequency'], conf['waveform_params']['duration'],
                                                conf['waveform_params']['start'], conf['source']['param_conversion_model'], conf['waveform_params']['is_only22'])
    det = detector.InterferometerList(conf['waveform_params']['ifo_list'])
    det.set_strain_data_from_zero_noise(
            sampling_frequency=conf['waveform_params']['sampling_frequency'],
            duration=conf['waveform_params']['duration'],
	    start_time=wf_gen.start_time,
    )
    det.inject_signal(waveform_generator=wf_gen, parameters=params)
    td_comp = 1e21 * np.vstack([ifo.time_domain_strain for ifo in det])

    fig, ax = plt.subplots(3, 1, figsize = (8, 10), sharex=True)
    for i in range(3):
        ax[i].plot(sample['d_t'][i], label='peri')
        ax[i].plot(td_comp[i], label='orig')
    ax[0].legend()
    #ax[0].set_xlim([5000, 6000])
    plt.savefig('sample_test.png')
    plt.close()


"""

A file containing plotting functions for the pycentricity package.

"""
import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import ticker
from matplotlib.ticker import ScalarFormatter
from matplotlib import rcParams

import corner
import pycentricity.waveform as wf
import pycentricity.overlap as ovlp

import wquantiles as wq
import bilby as bb

fontparams = {'mathtext.fontset': 'stix',
             'font.family': 'serif',
             'font.serif': "Times New Roman",
             'mathtext.rm': "Times New Roman",
             'mathtext.it': "Times New Roman:italic",
             'mathtext.sf': 'Times New Roman',
             'mathtext.tt': 'Times New Roman'}
rcParams.update(fontparams)

parameter_keys = dict(
    chirp_mass='$\mathcal{M}$ [M$_{\odot}$]',
    chirp_mass_source='$\mathcal{M}^\mathrm{source}$ [M$_{\odot}$]',
    mass_ratio='$q$',
    mass_1='$m_1$',
    mass_2='$m_2$',
    luminosity_distance='$d_\mathrm{L}$ [Mpc]',
    ra='RA',
    dec='DEC',
    chi_eff='$\chi_\mathrm{eff}$',
    chi_1='$\chi_1$',
    chi_2='$\chi_2$',
    chi_p='$\chi_p$',
    a_1='$a_1$',
    a_2='$a_2$',
    tilt_1='$\\theta_1$',
    tilt_2='$\\theta_2$',
    phi_12='$\phi_{12}$',
    phi_jl='$\phi_{jl}$',
    theta_jn='$\\theta_\mathrm{jn}$',
    phase='$\phi$',
    psi='$\psi$',
    log_eccentricity='log$_{10}(e_{10})$',
    geocent_time='$t_\mathrm{geo}$ [s]'
)

def get_data(data_list):
    """
    Simply transpose and stack a list of data.
    :param data_list: list
        list of data to transpose and stack
    :return:
        transposed and stacked list
    """
    return np.transpose(np.vstack(data_list))


def update_log_eccentricity_bins_and_range_for_corner(
    minimum_log_eccentricity, maximum_log_eccentricity, number_of_bins, subset_keys, subset
):
    bins = []
    range = []
    for k in subset_keys[subset]:
        bins.append(number_of_bins)
        if 'log_eccentricity' in k:
            range.append([minimum_log_eccentricity, maximum_log_eccentricity])
        else:
            range.append(0.9999)
    return bins, range


def plot_corner_weights_against_parameter(log_weights, other_parameter_array, parameter_name, label):
   data = get_data([log_weights, other_parameter_array])
   corner.corner(data, labels=['$ln w$', parameter_keys[parameter_name]])
   plt.savefig('{}_weight_{}_corner.png'.format(label, parameter_name), bbox_inches='tight')

def plot_corner_weights_eccentricities(log_weights, log_eccentricities, label):
   data = get_data([log_weights, log_eccentricities])
   corner.corner(data, labels=['$ln w$', '$log_{10}e$'])
   plt.savefig('{}_weight_eccentricity_corner.png'.format(label), bbox_inches='tight')
   
     
def plot_reweighted_posteriors(
        posterior, eccentricities, new_weights, label, original_weights=None, injection_values=None, subset='all',
):
    """
    Plot a set of posteriors with the original samples in turquoise and the reweighted posteriors in grey
    :param posterior: dictionary
        samples from a parameter estimation run
    :param new_weights: list
        list of weights, equal in length to the number of samples
    :param label: string
        string to use to label the output plot
    :param original_weights: list
        if needed, to denote original weights. useful if we need to get rid of certain samples
        (give them a weight of zero in both lists)_
    :param injection_values: dictionary
        optional dictionary of injected values
    """
    if original_weights is None:
        original_weights = [0.5] * len(new_weights)
    subset_keys = dict(
        all=['mass_1', 'mass_2', 'a_1', 'a_2', 'tilt_1', 'tilt_2', 'phi_12', 'phi_jl', 'log_eccentricity', 'chi_p', 'theta_jn', 'psi', 'ra', 'dec'],
        ecc_precc=['chirp_mass', 'chi_p', 'log_eccentricity'],
        O3a=['chirp_mass_source', 'mass_ratio', 'log_eccentricity', 'luminosity_distance', 'theta_jn', 'psi', 'phase', 'ra', 'dec', 'geocent_time'],
        intrinsic=['chirp_mass', 'mass_ratio', 'chi_1', 'chi_2', 'log_eccentricity'],
        source_frame_intrinsic=['chirp_mass_source', 'mass_ratio', 'chi_1', 'chi_2', 'log_eccentricity'],
        extrinsic=['luminosity_distance', 'theta_jn', 'psi', 'phase'],
        spins=['chi_1', 'chi_2', 'chi_eff'],
        S200114f=['chirp_mass_source', 'mass_ratio', 'chi_1', 'chi_2', 'log_eccentricity', 'luminosity_distance', 'ra', 'dec', 'geocent_time']
    )
    posterior['log_eccentricity'] = -10
    posterior.loc[0, 'log_eccentricity'] = -9
    parameters = [posterior[parameter] for parameter in subset_keys[subset]]
    labels = [parameter_keys[parameter] for parameter in subset_keys[subset]]
    for p, l in zip(parameters, labels):
        print(l, np.min(p), np.max(p))
    truths = None
    number_of_bins = 20
    bins, range = update_log_eccentricity_bins_and_range_for_corner(
                     -10, -8, number_of_bins, subset_keys, subset
                  )
    if injection_values is not None:
        truths = [injection_values[parameter] for parameter in subset_keys[subset]]
    figure = corner.corner(get_data(parameters), weights=original_weights, labels=labels,
                           bins=bins, smooth=0.9, label_kwargs=dict(fontsize=20), titles=True,
                           title_kwargs=dict(fontsize=20), color='darkturquoise', # color='#0072C1', 
                           range=range,
                           levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                           hist_kwargs=dict(density=True, lw=2),
                           plot_density=False, plot_datapoints=True, fill_contours=True, label='unweighted',
                           max_n_ticks=2)
    # Now add back in the real eccentricities
    posterior['log_eccentricity'] = np.log10(eccentricities)
    bins, range = update_log_eccentricity_bins_and_range_for_corner(
                     -3.68, np.log10(0.4), number_of_bins, subset_keys, subset
                  )
    parameters = [posterior[parameter] for parameter in subset_keys[subset]]
    corner.corner(get_data(parameters), weights=new_weights, bins=bins, labels=labels, fig=figure,
                  smooth=0.9, label_kwargs=dict(fontsize=20), titles=True,
                  title_kwargs=dict(fontsize=20), color='grey', truth_color='hotpink', #color='#228B22', truth_color='#FF8C00',
                  hist_kwargs=dict(density=True, lw=2), truths=truths,
                  range=range,
                  levels=(1 - np.exp(-0.5), 1 - np.exp(-2), 1 - np.exp(-9 / 2.)),
                  plot_density=False, plot_datapoints=True, fill_contours=True, label='weighted',
                  max_n_ticks=2)
    for ax in figure.get_axes():
        ax.tick_params(axis='both', labelsize=18, labelrotation=25)
    plt.savefig("{}_{}_corner.png".format(label, subset))
    print('saving figure {}'.format("{}_{}_corner.png".format(label, subset)))
    plt.clf()


def plot_chi_p_histogram(result, output_folder, label=''):
    bin_number = 60
    rwidth = 0.999
    posterior = result.posterior
    priors = result.priors
    print('number of samples: {}'.format(len(posterior['luminosity_distance'])))
    prior_samples = pd.DataFrame(priors.sample(len(posterior['luminosity_distance'])))
    #posterior_with_spins = bb.gw.conversion.generate_all_bbh_parameters(posterior)
    posterior_with_spins = posterior
    priors_with_spins = bb.gw.conversion.generate_all_bbh_parameters(prior_samples)
    chi_p = posterior_with_spins['chi_p']
    min = np.quantile(np.array(chi_p), 0.05)
    max = np.quantile(np.array(chi_p), 0.95)
    median = np.quantile(np.array(chi_p), 0.5)
    upper_bound = max - median
    lower_bound = median - min
    print('measured chi_p: {0:.2f} (+ {1:.2f}, - {2:.2f})'.format(median, upper_bound, lower_bound))
    print('number of chi_p samples: {}'.format(len(chi_p)))
    chi_p_prior = priors_with_spins['chi_p']
    # Plot the chi_p histogram
    fig = plt.subplots(figsize=(6, 3.5))
    prior_n, _, _ = plt.hist(chi_p_prior, bins=bin_number, align='left', color='darkturquoise', rwidth=rwidth, alpha=0.3, label='prior', histtype='stepfilled')
    post_n, b, _ = plt.hist(chi_p, bins=bin_number, align='left', color='grey', rwidth=rwidth, alpha=0.75, label=label)
    #plt.axvline(0, color='hotpink', alpha=1, label='injection', linewidth=2)
    plt.legend(fontsize=14, loc='upper left')
    plt.yticks(visible=False)
    plt.grid(False)
    plt.xlabel('$\chi_{\\rm p}$', fontsize=14)
    plt.ylabel('$p(\chi_{\\rm p}|d)$', fontsize=14, labelpad=15)
    plt.xlim([np.min(chi_p_prior), b[-2]])
    plt.ylim([0, np.max(post_n) + (np.max(post_n) / 3)])
    # Save the figure
    output_file = '{}/chi_p_histogram.pdf'.format(output_folder)
    print('output file: {}'.format(output_file))
    plt.savefig(output_file, bbox_inches='tight')


def plot_eccentricity_histogram(eccentricities, weights, injection, output_folder, label='', injected_eccentricity=0.1, minimum_log_eccentricity=-4, maximum_log_eccentricity=np.log10(0.2), nbins=40):
    """
    Plot a 1d eccentricity histogram, with 90% credible interval bounds if injection=False,
    or an injected value line if injection=True.
    :param eccentricities: list
        list of eccentricities
    :param injection: boolean
        whether the data is taken from an injection
    :param output_folder: string
        location of the output (where we will save the figure)
    :param weights: list
        list of weights
    :pram injected_eccentricity: float
        the value of the injection, if injection is True
    """
    # Compute things to put on the plot
    maximum_eccentricity = wq.quantile(np.array(eccentricities), np.array(weights), 0.9)
    minimum_eccentricity = wq.quantile(np.array(eccentricities), np.array(weights), 0.1)
    min = wq.quantile(np.array(eccentricities), np.array(weights), 0.05)
    max = wq.quantile(np.array(eccentricities), np.array(weights), 0.95)
    median = wq.quantile(np.array(eccentricities), np.array(weights), 0.5)
    upper_bound = max - median
    lower_bound = median - min
    print('e_max: {}, e_min: {}'.format(maximum_eccentricity, minimum_eccentricity))
    print('measured e: {0:.2f} (+ {1:.2f}, - {2:.2f})'.format(median, upper_bound, lower_bound))
    # Plot the eccentricity histogram
    bins = np.linspace(minimum_log_eccentricity, maximum_log_eccentricity, nbins)
    fig, ax = plt.subplots(figsize=(6, 3.5))
    # plot prior
    prior = []
    for ecc in bins[0:-1]:
        i = 0
        while i < int(len(eccentricities) / nbins):
            prior.append(ecc)
            i = i+1
    #n, _, _ = plt.hist(prior, bins=bins, align='left', color='darkturquoise', rwidth=0.999, alpha=0.3, label='prior', histtype='stepfilled')
    n, _, _ = plt.hist(np.log10(eccentricities), bins=nbins, color='grey', rwidth=0.999, alpha=0.75, weights=weights, label=label)
    ax.yaxis.set_major_formatter(ScalarFormatter())
    #if not injection:
        #plt.axvline(minimum_eccentricity, color='r', alpha=0.5)
        #plt.axvline(maximum_eccentricity, color='r', alpha=0.5)
    #else:
	#plt.axvline(injected_eccentricity, color='hotpink', alpha=1, label='injection', linewidth=2)
    plt.legend(fontsize=14)
    plt.yticks(visible=False)
    #plt.yscale('log')
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    plt.grid(False)
    plt.xlim([bins[0], np.max(np.log10(eccentricities))])
    plt.xlabel('$\log_{10}(e_{10})$', fontsize=14)
    plt.ylabel('$p(\log_{10}(e_{10})|d)$', fontsize=14, labelpad=15)
    # Save the figure
    output_file = '{}/{}_eccentricity_histogram.pdf'.format(output_folder, label)
    plt.savefig(output_file, bbox_inches='tight')
    return n, bins, _

def plot_2d_overlap(overlaps, time_grid_mesh, phase_grid_mesh):
    """
    Plot the 2D grid of time vs phase shifts.
    :param overlaps: 2D grid
        grid of calculated overlaps
    :param time_grid_mesh: 1D grid
        grid of proposed time shift
    :param phase_grid_mesh: 1D grid
        grid of proposed phase shifts
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    locator = ticker.LogLocator(base=np.exp(1))
    plt.contourf(phase_grid_mesh, np.divide(time_grid_mesh, 4096),
                 np.subtract(1, overlaps), locator=locator,
                 cmap='viridis_r')
    plt.ylabel("Time shift (s)")
    plt.xlabel("Phase shift")
    plt.colorbar()
    plt.title("1 - Overlap")
    plt.savefig("2d_overlap")
    plt.clf()


def plot_2d_heatmap_with_eccentricity(
        posterior_samples, parameter_name, weights, eccentricities, eccentricity_bins,
        injected_parameter=None, injected_eccentricity=None,
):
    """
    Plot a 2d heatmap of weighted posterior samples against eccentricity.
    :param posterior_samples: dictionary
        samples from a parameter estimation run
    :param parameter_name: string
        the name of the parameter, used to extract specific samples from the posterior samples
        and to label the plot
    :param injected_parameter: float
        value of the injected parameter
    :param injected_eccentricity: float
        value of the injected eccentricity
    :param weights: list
        list of weights for the posterior samples (will not be applied to the eccentricity data)
    :param eccentricities: list
        list of eccentricities
    :param eccentricity_bins: array
        bins into which to sort the eccentricities, i.e. np.logspace(-4, np.log10(0.2), 40)
    """
    number_of_bins = len(eccentricity_bins)
    hist_parameter, bin_edges_parameter = np.histogram(
        posterior_samples[parameter_name], bins=number_of_bins, weights=weights
    )
    hist_eccentricity, bin_edges_eccentricity = np.histogram(eccentricities, bins=eccentricity_bins)
    counts = [[p * e for e in hist_eccentricity] for p in hist_parameter]
    fig = plt.figure()
    plt.contourf(bin_edges_eccentricity[1:], bin_edges_parameter[1:], counts, cmap='Greys')
    plt.ylabel("reweighted {}".format(parameter_keys[parameter_name]), fontsize=14)
    plt.xlabel("eccentricity, $e$", fontsize=14)
    if injected_eccentricity is not None:
        plt.axvline(injected_eccentricity, linewidth=1, color='darkturquoise', label='injection')
        plt.axhline(injected_parameter, linewidth=1, color='darkturquoise')
        plt.scatter(injected_eccentricity, injected_parameter, color='darkturquoise')
        plt.legend(fontsize=14)
    plt.savefig("heatmap_injection_{}.pdf".format(parameter_name), bbox_inches='tight')
    plt.clf()


def plot_detector_strain_asd(ax, df, interferometer):
    asd = bb.gw.utils.asd_from_freq_series(
        freq_data=interferometer.strain_data.frequency_domain_strain, df=df)
    ax.loglog(interferometer.strain_data.frequency_array[interferometer.strain_data.frequency_mask],
              asd[interferometer.strain_data.frequency_mask],
              label=interferometer.name)
    ax.loglog(interferometer.strain_data.frequency_array[interferometer.strain_data.frequency_mask],
              interferometer.amplitude_spectral_density_array[interferometer.strain_data.frequency_mask],
              lw=1.0, label=interferometer.name + ' ASD')


def plot_response_to_waveform(ax, df, waveform_polarisations, parameters, label, interferometer):
    response = interferometer.get_detector_response(waveform_polarisations, parameters)
    signal_asd = bb.gw.utils.asd_from_freq_series(
        freq_data=response, df=df)
    line, = ax.loglog(interferometer.strain_data.frequency_array[interferometer.strain_data.frequency_mask],
                      signal_asd[interferometer.strain_data.frequency_mask],
                      label=label)
    return line,


def _layout_frequency_domain_strain_plot(fig, ax):
    ax.grid(True)
    ax.set_ylabel(r'Strain [strain/$\sqrt{\rm Hz}$]')
    ax.set_xlabel(r'Frequency [Hz]')
    ax.legend(loc='best')
    fig.tight_layout()


def plot_detector_response(signals, parameters, labels, interferometer, show=True):
    fig, ax = plt.subplots()
    df = interferometer.strain_data.frequency_array[1] - interferometer.strain_data.frequency_array[0]
    plot_detector_strain_asd(ax, df, interferometer)
    for signal, label in zip(signals, labels):
        line, = plot_response_to_waveform(ax, df, signal, parameters, label, interferometer)
    _layout_frequency_domain_strain_plot(fig, ax)
    fig.savefig('{}_frequency_domain_data.png'.format(interferometer.name))
    if show:
        print(
            'displaying frequency domain strain for interferometer {} with labels {}'.format(
                interferometer.name, labels
            )
        )
        plt.show()
    plt.close(fig)


def animate_detector_response(
        eccentric_signal, circular_signal, eccentric_label, circular_label,
        eccentricities, parameters, interferometer,
        circular_waveform_generator, minimum_frequency, show=False,
):
    fig, ax = plt.subplots()
    df = interferometer.strain_data.frequency_array[1] - interferometer.strain_data.frequency_array[0]
    plot_detector_strain_asd(ax, df, interferometer)
    plot_response_to_waveform(ax, df, circular_signal, parameters, circular_label, interferometer)
    line, = plot_response_to_waveform(ax, df, eccentric_signal, parameters, eccentric_label, interferometer)
    _layout_frequency_domain_strain_plot(fig, ax)

    def init():
        line.set_ydata([np.nan]
                       * len(interferometer.strain_data.frequency_array[interferometer.strain_data.frequency_mask])
                       )
        return line,

    def animate(i):
        parameters['eccentricity'] = eccentricities[i]
        new_eccentric_waveform = wf.seobnre_bbh_with_spin_and_eccentricity(
            parameters, interferometer.sampling_frequency, minimum_frequency
        )
        eccentric_waveform, eccentric_signal, max_overlap, time_shift, phase_shift = ovlp.maximise_overlap(
            new_eccentric_waveform,
            circular_signal,
            circular_waveform_generator.sampling_frequency,
            circular_waveform_generator.frequency_array,
            interferometer.power_spectral_density,
            minimum_frequency=interferometer.minimum_frequency,
            maximum_frequency=interferometer.maximum_frequency
        )
        eccentric_response = interferometer.get_detector_response(eccentric_signal, parameters)
        seobnre_asd = bb.gw.utils.asd_from_freq_series(
            freq_data=eccentric_response, df=df)
        plt.title('{}: e={}, overlap={}'.format(interferometer.name, eccentricities[i], max_overlap))
        line.set_ydata(seobnre_asd[interferometer.strain_data.frequency_mask])  # update the data.
        return line,

    ani = animation.FuncAnimation(
        fig, animate, init_func=init, interval=1, blit=False, frames=len(eccentricities))
    ani.save('animate_{}.gif'.format(interferometer.name), writer=matplotlib.animation.PillowWriter(fps=10))
    if show:
        plt.show()
    plt.close(fig)

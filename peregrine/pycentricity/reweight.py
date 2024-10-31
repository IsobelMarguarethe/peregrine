"""

A file for reweighting non-eccentric results using an eccentric waveform for the pycentricity package.

"""
import matplotlib.pyplot as plt

import bilby as bb
import numpy as np
import pycentricity.overlap as ovlp
import pycentricity.waveform as wf
import numpy.random as random
from scipy.interpolate import interp1d
from scipy.special import i0e, logsumexp

import subprocess as sp


def calculate_log_weight(eccentric_log_likelihood_array, circular_log_likelihood):
    """
    Return the log weight for the sample with the passed-in eccentric likelihood array.
    :param eccentric_log_likelihood_array: array/list
        list of eccentric log-likelihoods
    :param circular_log_likelihood: float
        value of the circular log-likelihood
    :return:
        average_log_weight: float
            the log weight for the sample
    """
    individual_log_weight = [ll - circular_log_likelihood for ll in eccentric_log_likelihood_array]
    average_weight = np.mean(np.exp(individual_log_weight))
    average_log_weight = np.log(average_weight)
    return average_log_weight


def pick_weighted_random_eccentricity(cumulative_density_grid, eccentricity_grid):
    """
    Return a random eccentricity, weighted by a cumulative density function.
    :param cumulative_density_grid: array
        1D grid of cumulative densities
    :param eccentricity_grid: array
        1D grid of eccentricities
    :return:
        random_eccentricity: float
            the weighted random eccentricity chosen
    """
    interpolated_function = interp1d(cumulative_density_grid, eccentricity_grid)
    randsamp = random.random_sample()
    if randsamp < min(cumulative_density_grid):
        return (eccentricity_grid[1] - eccentricity_grid[0]) * random.random_sample() + eccentricity_grid[0]
    else:
        return interpolated_function(randsamp)


def cumulative_density_function(log_likelihood_grid):
    """
    Return a cumulative density function over a grid of eccentricities.
    :param log_likelihood_grid: array
        1D grid of log likelihoods
    :return:
    cumulative_density: array
        1D grid of cumulative densities
    """
    # Deal with extremely high values of log likelihood
    maximum_log_likelihood = np.max(log_likelihood_grid)
    # Ratio of likelihood to maximum log likelihood
    log_likelihood_grid = log_likelihood_grid - maximum_log_likelihood
    likelihood_grid = np.exp(log_likelihood_grid)
    cumulative_density =  np.cumsum(likelihood_grid)
    cumulative_density_normalised = cumulative_density / cumulative_density[-1]
    return cumulative_density_normalised


def deal_with_no_waveform_generation(label, eccentricity, intermediate_outfile):
    print('No waveform generated; disregard sample {}'.format(label))
    intermediate_outfile.write("{}\t\t{}\t\t{}\n".format(eccentricity, None, None))
    disregard = True
    return disregard


def obtain_output_parameters(log_likelihood_grid, eccentricity_grid, original_log_likelihood):
    cumulative_density_grid = cumulative_density_function(log_likelihood_grid)
    # We want to pick a weighted random point from within the CDF
    new_e = pick_weighted_random_eccentricity(cumulative_density_grid, eccentricity_grid)
    # Also return average log-likelihood
    average_log_likelihood = np.log(np.sum(np.exp(log_likelihood_grid)) / len(log_likelihood_grid))
    # Weight calculated using average likelihood
    log_weight = calculate_log_weight(log_likelihood_grid, original_log_likelihood)
    return new_e, average_log_likelihood, log_weight 


def set_up_intermediate_output_file(label, parameters):
    intermediate_outfile = open("{}_eccentricity_result.txt".format(label), "w")
    intermediate_outfile.write("sample parameters:\n")
    for key in parameters.keys():
        intermediate_outfile.write("{}:\t{}\n".format(key, parameters[key]))
    intermediate_outfile.write("\n-------------------------\n")
    intermediate_outfile.write("e\t\tlog_L\t\tmaximised_overlap\n")
    return intermediate_outfile


def new_weight(parameters, comparison_log_likelihood_value,
               eccentric_likelihood,
               minimum_log_eccentricity, maximum_log_eccentricity, number_of_eccentricity_bins,
               label, ecc_scale='log', fixed_eccentricity=None, verbose=False
    ):
    if fixed_eccentricity == None:
        eccentricity_grid = np.logspace(
            minimum_log_eccentricity, maximum_log_eccentricity, number_of_eccentricity_bins
        )
        if ecc_scale == 'linear':
            eccentricity_grid = np.linspace(
                10. ** minimum_log_eccentricity, 10. ** maximum_log_eccentricity, number_of_eccentricity_bins
            )
    else:
        eccentricity_grid = [fixed_eccentricity]

    log_likelihood_grid = []
    for e in eccentricity_grid:
        p = parameters
        p['eccentricity'] = e
        # IRS temporary - store the most recent parameters
        if verbose:
            set_up_intermediate_output_file(label, p)
        # Calculate eccentric log likelihood
        eccentric_likelihood.parameters = p
        eccentric_log_L = eccentric_likelihood.log_likelihood_ratio()
        # Store log likelihood
        log_likelihood_grid.append(eccentric_log_L)
    return obtain_output_parameters(log_likelihood_grid, eccentricity_grid, comparison_log_likelihood_value)


def recalculated_log_l_differs_from_original_significantly_warning(recalculated_log_likelihood, log_L):
    # Print a warning if this is much different to the likelihood stored in the results
    print('original log L: {}'.format(log_L))
    print('recalculated log L: {}'.format(recalculated_log_likelihood))
    if abs(recalculated_log_likelihood - log_L) / log_L > 0.1:
        percentage = abs(recalculated_log_likelihood - log_L) / log_L * 100
        print(
            "WARNING :: recalculated log likelihood differs from original by {}%".format(
                percentage
            )
        )
        print('original log L: {}'.format(log_L))
        print('recalculated log L: {}'.format(recalculated_log_likelihood))



def output_reweighting_progress(i, number_of_samples):
    # IRS TODO docstring
    print(
            "new weight calculation {}% complete".format(
                np.round((i + 1) / number_of_samples * 100, 2)
         )
    )


def write_results_to_output(i, eccentricity, new_log_L, log_weight, output, outfile):
    # IRS TODO docstring
    outfile.write("{}\t\t{}\t\t{}\t\t{}\n".format(i, eccentricity, new_log_L, log_weight))
    output["eccentricity"].append(eccentricity)
    output["new_log_L"].append(new_log_L)
    output["log_weight"].append(log_weight)


def set_up_outfile(output_folder, label):
    # IRS TODO docstring
    outfile = open("{}/{}_master_output_store.txt".format(output_folder, label), "w")
    outfile.write("i\t\te\t\tnew_log_L\t\tlog_w\n")
    return outfile


def reweight_by_eccentricity(samples, minimum_log_eccentricity, maximum_log_eccentricity, number_of_eccentricity_bins,
                             eccentric_likelihood, comparison_likelihood,
                             label, output_folder, check_every_sample=False, sub_result_file=None,
                             reference_frequency=None, sampling_frequency=None, ecc_scale='log', verbose=False):

    output = {key: [] for key in ["eccentricity", "new_log_L", "log_weight"]}
    # Write the output file along the way
    outfile = set_up_outfile(output_folder, label)

    number_of_samples = len(samples['mass_1'])

    for i in range(number_of_samples):
        samples_i = {key: samples[key][i] for key in samples.keys()}
        if verbose:
            print(i, samples_i)
        comparison_likelihood.parameters = samples_i
        comparison_log_likelihood_value = comparison_likelihood.log_likelihood_ratio()

        eccentricity, new_log_L, log_weight = new_weight(samples_i,
               comparison_log_likelihood_value, eccentric_likelihood,
               minimum_log_eccentricity, maximum_log_eccentricity, number_of_eccentricity_bins,
               label, ecc_scale, verbose=verbose)
        if verbose:
            print(eccentricity, new_log_L, log_weight)
        write_results_to_output(i, eccentricity, new_log_L, log_weight, output, outfile)
        output_reweighting_progress(i, number_of_samples)
    outfile.close()
    return output


def reweight_by_fixed_eccentricity(samples, fixed_eccentricity, eccentric_waveform_generator, comparison_waveform_generator,
                                   eccentric_likelihood, comparison_likelihood,
                                   label, output_folder, check_every_sample=False, sub_result_file=None,
                                   reference_frequency=None, sampling_frequency=None, ecc_scale='log'):

    converted_samples = get_converted_samples(samples)
    parameter_list = get_parameter_list(converted_samples, np.log10(fixed_eccentricity))

    # Generate the IMRPhenomD waveform for each sample
    comparison_waveform_strain_list = [
        comparison_waveform_generator.frequency_domain_strain(p)
        for p in parameter_list
    ]

    output = {key: [] for key in ["eccentricity", "new_log_L", "log_weight"]}

    # Write the output file along the way
    outfile = set_up_outfile(output_folder, label)

    number_of_samples = len(comparison_waveform_strain_list)


    for i, comparison_waveform in enumerate(comparison_waveform_strain_list):

        comparison_likelihood.parameters = parameter_list[i]
        comparison_log_likelihood_value = comparison_likelihood.log_likelihood_ratio()
        eccentricity, new_log_L, log_weight = new_weight(parameter_list[i],
               comparison_waveform, comparison_log_likelihood_value,
               eccentric_waveform_generator, eccentric_likelihood,
               np.log10(fixed_eccentricity), np.log10(fixed_eccentricity), 1,
               label, ecc_scale, fixed_eccentricity)
        write_results_to_output(i, eccentricity, new_log_L, log_weight, output, outfile)
        output_reweighting_progress(i, number_of_samples)
    outfile.close()
    return output

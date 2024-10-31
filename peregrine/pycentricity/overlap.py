"""

A file for maximising overlaps for the pycentricity package.

"""

import numpy as np
import bilby as bb


def maximised_overlap(a, b, frequency, PSD, lower_freq, upper_freq):
    psd_interp = PSD.power_spectral_density_interpolated(frequency)
    df = frequency[1] - frequency[0]
    duration = 1. / df
    li = int(lower_freq / df)
    ui = int(upper_freq / df)
    inner_a = bb.gw.utils.noise_weighted_inner_product(
        a['plus'][li:ui], a['plus'][li:ui], psd_interp[li:ui], duration).real
    inner_a += bb.gw.utils.noise_weighted_inner_product(
        a['cross'][li:ui], a['cross'][li:ui], psd_interp[li:ui], duration).real
    inner_b = bb.gw.utils.noise_weighted_inner_product(
        b['plus'][li:ui], b['plus'][li:ui], psd_interp[li:ui], duration).real
    inner_b += bb.gw.utils.noise_weighted_inner_product(
        b['cross'][li:ui], b['cross'][li:ui], psd_interp[li:ui], duration).real
    inner_ab_of_t = 4 / duration * np.fft.fft(
        a['plus'][li:ui][:-1] * np.conj(b['plus'][li:ui][:-1]) / psd_interp[li:ui][:-1])
    inner_ab_of_t += 4 / duration * np.fft.fft(
        a['cross'][li:ui][:-1] * np.conj(b['cross'][li:ui][:-1]) / psd_interp[li:ui][:-1])
    times = np.linspace(0, duration, len(inner_ab_of_t) + 1)[:-1]
    max_idx = np.argmax(abs(inner_ab_of_t))
    max_time = - times[max_idx]
    max_phase = - np.angle(inner_ab_of_t[max_idx]) / 2
    max_overlap = max(abs(inner_ab_of_t)) / np.sqrt(inner_a * inner_b)
    return max_overlap, max_time, max_phase

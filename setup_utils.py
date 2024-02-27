"""Dictionary generating functions for setting up a simulation"""

import numpy as np

def define_parameters(
    population_size,
    angular_samples,
    cognitive_coeff,
    social_coeff,
    intertia_weight,
    max_steps,
    static_targets,
    beamwidth,
    sidelobe_suppression,
    max_particle_velocity,
    neighbourhood_size,
):
    phi = np.linspace(-np.pi, np.pi, angular_samples)
    targets = ((np.asarray(static_targets) / (2 * np.pi)) + 0.5) * angular_samples - 1
    targets = targets.astype(int)
    beamwidth_in_samples = np.asarray(beamwidth * angular_samples / 2 * np.pi)
    beamwidth_in_samples = beamwidth_in_samples.astype(int)
    return {
        "population_size": population_size,
        "samples": angular_samples,
        "phi": phi,
        "cognitive_coeff": cognitive_coeff,
        "social_coeff": social_coeff,
        "intertia_weight": intertia_weight,
        "max_steps": max_steps,
        "targets": targets,
        "beamwidth": beamwidth,
        "beamwidth_samples": beamwidth_in_samples,
        "sidelobe_suppression": sidelobe_suppression,
        "max_particle_velocity": max_particle_velocity,
        "neighbourhood_size": neighbourhood_size,
    }


def define_logging(show_plots, plots_persist, verbose):
    return {
        "show_plots": show_plots,
        "plots_persist": plots_persist,
        "verbose": verbose,
    }


def define_ULA(frequency, spacing_coeff, num_elements):
    wavelength = 3e9 / frequency
    spacing = wavelength * spacing_coeff
    N = (num_elements - 1) / 2
    positions = np.linspace(spacing * -N, spacing * N, num_elements)
    return {
        "num_elements": num_elements,
        "spacing": spacing,
        "positions": positions,
        "frequency": frequency,
        "wavelength": wavelength,
        "wavenumber": 2 * np.pi / wavelength,
    }

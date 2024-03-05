"""Dataclasses for setting up simulation"""

import numpy as np
from dataclasses import dataclass


@dataclass
class Logging:
    show_plots: bool = False
    plots_persist: bool = False
    verbose: bool = False


class Parameters:
    def __init__(
        self,
        population_size,
        samples,
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
        self.population_size = population_size
        self.samples = samples
        self.phi = np.linspace(-np.pi, np.pi, samples)
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.intertia_weight = intertia_weight
        self.max_steps = max_steps
        self.targets = (
            ((np.asarray(static_targets) / (2 * np.pi)) + 0.5) * samples - 1
        ).astype(int)
        self.beamwidth = beamwidth
        self.beamwidth_samples = np.asarray(beamwidth * samples / 2 * np.pi, dtype=int)
        self.sidelobe_suppression = sidelobe_suppression
        self.max_particle_velocity = max_particle_velocity
        self.neighbourhood_size = neighbourhood_size


class ULA:
    def __init__(self, frequency, spacing_coeff, num_elements):
        self.num_elements = num_elements
        self.frequency = frequency
        self.wavelength = 3e9 / frequency
        self.spacing = self.wavelength * spacing_coeff
        N = (num_elements - 1) / 2
        self.positions = np.linspace(self.spacing * -N, self.spacing * N, num_elements)
        self.wavenumber = 2 * np.pi / self.wavelength

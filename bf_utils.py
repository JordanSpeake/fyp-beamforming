import numpy as np
from dataclasses import dataclass


@dataclass
class Logging:
    show_plots: bool = False
    plots_persist: bool = False
    verbose: bool = False
    write_results: bool = False
    use_uniform_particle: bool = False


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
        max_particle_velocity,
        neighbourhood_size,
        num_tiles,
        phase_bit_depth,
    ):
        self.population_size = population_size
        self.samples = samples  # the number of samples taken when calculating the array factor in each direction
        self.theta = np.linspace(-np.pi / 2, np.pi / 2, samples)
        self.phi = np.linspace(-np.pi / 2, np.pi / 2, samples)
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.intertia_weight = intertia_weight
        self.max_steps = max_steps
        self.targets = (
            ((np.asarray(static_targets, dtype=float) / (np.pi)) + 0.5) * samples - 1
        ).astype(int)
        self.max_particle_velocity = max_particle_velocity
        self.neighbourhood_size = neighbourhood_size
        self.num_tiles = num_tiles
        self.phase_bit_depth = phase_bit_depth

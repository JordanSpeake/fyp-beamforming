import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt

def random_complex(size):
    """Generates a random complex number, uniformly sampled from a zero-centred unit circle"""
    return np.sqrt(np.random.uniform(0, 1, size)) * np.exp(
        1.0j * np.random.uniform(0, 2 * np.pi, size)
    )


def spherical_to_uv(spherical_coords):
    """Convert [theta, phi] to [u, v]. From spherical to directional cosine coordinates."""
    theta = spherical_coords[0]
    phi = spherical_coords[1]
    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)
    return np.asarray([u, v])


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
        elitism_count,
        elitism_replacement_chance,
    ):
        self.population_size = population_size
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff
        self.intertia_weight = intertia_weight
        self.max_steps = max_steps
        self.max_particle_velocity = max_particle_velocity
        self.neighbourhood_size = neighbourhood_size
        self.targets = static_targets
        self.num_tiles = num_tiles
        self.phase_bit_depth = phase_bit_depth
        self.samples = samples
        self.u_grid, self.v_grid = np.meshgrid(np.linspace(-1, 1, samples), np.linspace(-1, 1, samples))
        self.elitism_count = elitism_count
        self.elitism_replacement_chance = elitism_count

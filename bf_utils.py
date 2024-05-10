import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt


def to_dB(value):
    return 10 * np.log10(value)

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

def calculate_mean_squared_error(input_1, input_2):
    error = input_1 - input_2
    squared_error = np.power(error, 2)
    mean_squared_error = np.mean(squared_error)
    return mean_squared_error


def quantize(value, bit_depth):
    bits = np.power(2, bit_depth)
    quantisation_step = int((value / 2 * np.pi) * bits)
    value = quantisation_step * 2 * np.pi / bits
    return value

@dataclass
class Logging:
    show_plots: bool = False
    plots_persist: bool = False
    verbose: bool = False
    write_results: bool = False
    output_final_values: bool = False


class Parameters:
    def __init__(
        self,
        samples,
        max_steps,
        max_particle_velocity,
        phase_bit_depth,
        num_clusters,
        subswarm_size,
        subswarm_init_radius,
        num_subswarms,
        subswarm_charge,
        centroid_velocity_coeff,
        particle_inertia_weight,
        dois,
        dnois,
        mse_target_levels,
        rerandomisation_proximity,
        social_coeff,
    ):
        self.subswarm_init_radius = subswarm_init_radius
        self.num_clusters = num_clusters
        self.subswarm_size = subswarm_size
        self.num_subswarms = num_subswarms
        self.max_steps = max_steps
        self.max_particle_velocity = max_particle_velocity
        self.phase_bit_depth = phase_bit_depth
        self.samples = samples
        self.u_grid, self.v_grid = np.meshgrid(np.linspace(-1, 1, samples), np.linspace(-1, 1, samples))
        self.w_grid = np.zeros_like(self.u_grid)
        self.w_grid = np.emath.sqrt(1 - np.power(self.u_grid, 2) - np.power(self.v_grid, 2))
        self.w_grid = self.w_grid.real
        self.subswarm_charge = subswarm_charge
        self.centroid_velocity_coeff = centroid_velocity_coeff
        self.particle_inertia_weight = particle_inertia_weight
        self.dois = dois
        self.dnois = dnois
        self.rerandomisation_proximity = rerandomisation_proximity
        self.social_coeff = social_coeff
        self.mse_target_levels = mse_target_levels

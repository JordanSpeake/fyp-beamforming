from numba import jit
import numpy as np

@jit
def rpa_radiated_power(complex_weights, electric_field_preallocated, u_grid, v_grid, num_el_x, num_el_y, wavenumber, spacing):
    """Calculate the radiated_power for the given complex weights, returned in spherical coordinates (theta, phi)"""
    phases = np.angle(complex_weights)
    weights = np.abs(complex_weights)
    electric_field = electric_field_preallocated
    for m in range(num_el_x):
        for n in range(num_el_y):
            element = m * num_el_y + n
            exponent = phases[element] + 1j * wavenumber * (
                m * spacing[0] * u_grid + n * spacing[1] * v_grid
            )
            electric_field += weights[element] * np.exp(exponent)
    radiated_power = np.power(np.abs(electric_field), 2)
    return radiated_power

"""Module containing class definitions of implemented antenna types"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec

class Antenna:
    """Base class for all antennas"""
    def __init__(self, frequency, num_elements, parameters):
        self.frequency = frequency
        wavelength = 3e9 / frequency
        self.wavenumber = 2 * np.pi / wavelength
        self.num_elements = num_elements
        self.theta = np.tile(np.atleast_2d(parameters.theta), (parameters.samples, 1))
        self.phi = np.tile(np.atleast_2d(parameters.phi).T, (1, parameters.samples))
        self.theta_samples = parameters.theta
        self.phi_samples = parameters.phi
        self.sin_u = np.sin(self.theta) * np.cos(self.phi)
        self.sin_v = np.sin(self.theta) * np.sin(self.phi)
        self.figure = plt.figure()
        grid_spec = gridspec.GridSpec(2, 2)
        self.ax_tile_pattern = self.figure.add_subplot(grid_spec[:, 1])
        self.ax_untiled = self.figure.add_subplot(grid_spec[0, 0], projection="3d")
        self.ax_tiled = self.figure.add_subplot(grid_spec[1, 0], projection="3d")

    def reset_axes(self):
        """Reset and clear axes for the next step's data output to be plotted"""
        self.ax_tiled.clear()
        self.ax_tiled.set_title("ORIGINAL: Array Factor")
        self.ax_tiled.set_xlabel("Angle (Theta)")
        self.ax_tiled.set_ylabel("Angle (Phi)")
        self.ax_tiled.set_zlabel("Array Factor (dB)")
        self.ax_untiled.clear()
        self.ax_untiled.set_title("TILED: Array Factor")
        self.ax_untiled.set_xlabel("Angle (Theta)")
        self.ax_untiled.set_ylabel("Angle (Phi)")
        self.ax_untiled.set_zlabel("Array Factor (dB)")

    def display(
        self,
        untiled_weights,
        tiled_weights,
        tile_labels,
        persist=False,
        pause_time=0.01,
    ):
        """Display the given untiled and tiled weights, including the selection pattern"""

        untiled_af = self.array_factor(untiled_weights)
        tiled_af = self.array_factor(tiled_weights)
        R, P = np.meshgrid(self.phi_samples, self.theta_samples)
        X, Y = R * np.cos(P), R * np.sin(P)

        self.reset_axes()
        self.ax_untiled.plot_surface(X, Y, untiled_af)
        self.ax_tiled.plot_surface(X, Y, tiled_af)
        self.update_tiling_plot(tile_labels)
        if persist:
            plt.show()
        else:
            plt.pause(pause_time)

    def fitness(self, element_complex_weights, parameters):
        """Calculates the fitness of the given complex weights for this antenna"""
        score = 0
        for target in parameters.targets:
            score += self.array_factor_single(
                element_complex_weights, target[0], target[1]
            )
        return score


    def update_tiling_plot(self, tile_labels):
        #pylint: disable=unused-argument
        assert False, "update_tiling_plot() must be defined in child class."

    def array_factor_single(self, element_complex_weights, theta, phi):
        #pylint: disable=unused-argument
        assert False, "array_factor_single() must be defined in child class."

    def array_factor(self, element_complex_weights):
        #pylint: disable=unused-argument
        assert False, "array_factor() must be defined in child class."


class RectangularPlanar(Antenna):
    def __init__(self, frequency: float, spacing: tuple, num_el: tuple, parameters):
        num_elements = num_el[0] * num_el[1]
        self.num_el_x = num_el[0]
        self.num_el_y = num_el[1]
        self.spacing = spacing
        super().__init__(frequency, num_elements, parameters)

    def array_factor(self, element_complex_weights):
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        array_factor = np.zeros_like(self.theta, dtype=complex)
        for m in range(self.num_el_x):
            for n in range(self.num_el_y):
                element = m * self.num_el_y + n
                exponent = phases[element] + 1j * self.wavenumber * (
                    m * self.spacing[0] * self.sin_u + n * self.spacing[1] * self.sin_v
                )
                array_factor += weights[element] * np.exp(exponent)
        array_factor = 20 * np.log10(np.abs(array_factor))
        return array_factor

    def array_factor_single(self, element_complex_weights, theta, phi):
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        array_factor = 0 + 0j
        sin_u = np.sin(theta) * np.cos(phi)
        sin_v = np.sin(theta) * np.sin(phi)
        for m in range(self.num_el_x):
            for n in range(self.num_el_y):
                element = m * self.num_el_y + n
                exponent = phases[element] + 1j * self.wavenumber * (
                    m * self.spacing[0] * sin_u + n * self.spacing[1] * sin_v
                )
                array_factor += weights[element] * np.exp(exponent)
        array_factor = 20 * np.log10(np.abs(array_factor))
        return array_factor

    def update_tiling_plot(self, tile_labels):
        self.ax_tile_pattern.clear()
        self.ax_tile_pattern.set_xticks(
            np.arange(start=0, stop=self.num_el_x + 1, step=1)
        )
        self.ax_tile_pattern.set_yticks(
            np.arange(start=0, stop=self.num_el_y + 1, step=1)
        )
        self.ax_tile_pattern.grid(True)
        colors = list(mcolors.TABLEAU_COLORS)
        element_count = self.num_el_x * self.num_el_y
        for element in range(element_count):
            tile_group = tile_labels[element]
            x, y = np.divmod(element, self.num_el_x)
            self.ax_tile_pattern.fill_between(
                [x, x + 1], y, y - 1, color=colors[tile_group]
            )


class UniformLinear(Antenna):
    def __init__(self, frequency: float, spacing: float, num_el: int, parameters):
        self.spacing = spacing
        super().__init__(frequency, num_el, parameters)

    def array_factor(self, element_complex_weights):
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        array_factor = np.zeros_like(self.theta, dtype=complex)
        for element in range(self.num_elements):
            exponent = phases[element] + (
                1j * self.wavenumber * (element * self.spacing * self.sin_u)
            )
            array_factor += weights[element] * np.exp(exponent)
        array_factor = 20 * np.log10(np.abs(array_factor))
        return array_factor

    def array_factor_single(self, element_complex_weights, theta, phi):
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        array_factor = 0 + 0j
        exponent_constant = self.spacing * np.sin(theta) * np.cos(phi)
        for element in range(self.num_elements):
            exponent = phases[element] + (
                1j * self.wavenumber * (element * exponent_constant)
            )
            array_factor += weights[element] * np.exp(exponent)
        array_factor = 20 * np.log10(np.abs(array_factor))
        return array_factor

    def update_tiling_plot(self, tile_labels):
        self.ax_tile_pattern.clear()
        self.ax_tile_pattern.set_xticks(
            np.arange(start=0, stop=self.num_elements + 1, step=1)
        )
        self.ax_tile_pattern.grid(True)
        colors = list(mcolors.TABLEAU_COLORS)
        for element in range(self.num_elements):
            tile_group = tile_labels[element]
            x, y = np.divmod(element, self.num_elements)
            self.ax_tile_pattern.fill_between(
                [x, x + 1], y, y - 1, color=colors[tile_group]
            )


class Circular(Antenna):
    def __init__(self, frequency, radius, num_elements, parameters):
        self.radius = radius
        super().__init__(frequency, num_elements, parameters)

    def array_factor(self, element_complex_weights):
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        array_factor = np.zeros_like(self.theta, dtype=complex)
        for k in range(self.num_elements):
            element_angle = 2 * np.pi * k / self.num_elements
            exponent = phases[k] + 1j * self.wavenumber * (
                self.radius
                * (
                    np.sin(element_angle) * self.sin_u
                    + np.cos(element_angle) * self.sin_v
                )
            )
            array_factor += weights[k] * np.exp(exponent)
        array_factor = 20 * np.log10(np.abs(array_factor))
        return array_factor

    def array_factor_single(self, element_complex_weights, theta, phi):
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        sin_u = np.sin(theta) * np.cos(phi)
        sin_v = np.sin(theta) * np.sin(phi)
        array_factor = 0 + 0j
        for k in range(self.num_elements):
            element_angle = 2 * np.pi * k / self.num_elements
            exponent = phases[k] + 1j * self.wavenumber * (
                self.radius
                * (np.sin(element_angle) * sin_u + np.cos(element_angle) * sin_v)
            )
            array_factor += weights[k] * np.exp(exponent)
        array_factor = 20 * np.log10(np.abs(array_factor))
        return array_factor

    def update_tiling_plot(self, tile_labels):
        self.ax_tile_pattern.clear()
        self.ax_tile_pattern.set_xticks(
            np.arange(start=0, stop=self.num_elements + 1, step=1)
        )
        self.ax_tile_pattern.grid(True)
        colors = list(mcolors.TABLEAU_COLORS)
        for element in range(self.num_elements):
            tile_group = tile_labels[element]
            x, y = np.divmod(element, self.num_elements)
            self.ax_tile_pattern.fill_between(
                [x, x + 1], y, y - 1, color=colors[tile_group]
            )

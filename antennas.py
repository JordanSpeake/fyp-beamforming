"""Module containing class definitions of implemented antenna types"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches

class Antenna:
    """Base class for all antennas"""
    def __init__(self, frequency, num_elements, parameters):
        self.frequency = frequency
        wavelength = 3e9 / frequency
        self.wavenumber = 2 * np.pi / wavelength
        self.num_elements = num_elements
        self.samples = parameters.samples
        self.u_grid = parameters.u_grid
        self.v_grid = parameters.v_grid
        self.figure = plt.figure()
        grid_spec = gridspec.GridSpec(2, 2)
        self.ax_tile_pattern = self.figure.add_subplot(grid_spec[:, 1])
        self.ax_untiled = self.figure.add_subplot(grid_spec[0, 0])
        self.ax_tiled = self.figure.add_subplot(grid_spec[1, 0])
        self.targets = parameters.targets

    def update_array_factor_axis(self, axis, array_factor):
        """Reset and clear axes for the next step's data output to be plotted"""
        unit_circle = patches.Circle((0, 0), 1, color='b', fill=False)
        axis.clear()
        axis.set_xlabel("U")
        axis.set_ylabel("V")
        axis.imshow(array_factor, cmap='seismic', interpolation='nearest', extent=[-1, 1, -1, 1])
        axis.add_patch(unit_circle)

    def display(
        self,
        untiled_weights,
        tiled_weights,
        tile_labels,
        persist=False,
        pause_time=0.1,
    ):
        """Display the given untiled and tiled weights, including the selection pattern"""
        self.update_array_factor_axis(self.ax_untiled, self.array_factor(untiled_weights))
        self.ax_tiled.set_title("TILED: Array Factor")
        self.ax_untiled.set_title("UNTILED: Array Factor")
        self.update_array_factor_axis(self.ax_tiled, self.array_factor(tiled_weights))
        self.update_tiling_plot(tile_labels)
        if persist:
            plt.show()
        else:
            plt.pause(pause_time)

    def polar_to_uv(self, polar_coords):
        """Convert [theta, phi] to [u, v]. From spherical to directional cosine"""
        theta = polar_coords[0]
        phi = polar_coords[1]
        u = np.sin(theta) * np.cos(phi)
        v = np.sin(theta) * np.sin(phi)
        return np.asarray([u, v])

    def fitness(self, element_complex_weights, parameters):
        """Calculates the fitness of the given complex weights for this antenna"""
        score = 0
        for target in parameters.targets:
            target_uv = self.polar_to_uv(target)
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
        """Calculate the array factor for the given complex weights, returned in spherical coordinates (theta, phi)"""
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        array_factor = np.zeros((self.samples, self.samples), dtype=complex)
        for m in range(self.num_el_x):
            for n in range(self.num_el_y):
                element = m * self.num_el_y + n
                exponent = phases[element] + 1j * self.wavenumber * (
                    m * self.spacing[0] * self.u_grid + n * self.spacing[1] * self.v_grid
                )
                array_factor += weights[element] * np.exp(exponent)
        array_factor = 20 * np.log10(np.abs(array_factor))
        return array_factor

    def array_factor_single(self, element_complex_weights, u, v):
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        array_factor = 0 + 0j
        for m in range(self.num_el_x):
            for n in range(self.num_el_y):
                element = m * self.num_el_y + n
                exponent = phases[element] + 1j * self.wavenumber * (
                    m * self.spacing[0] * u + n * self.spacing[1] * v
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
        array_factor = np.zeros((self.samples, self.samples), dtype=complex)
        for element in range(self.num_elements):
            exponent = phases[element] + (
                1j * self.wavenumber * (element * self.spacing * self.u_grid)
            )
            array_factor += weights[element] * np.exp(exponent)
        array_factor = 20 * np.log10(np.abs(array_factor))
        return array_factor

    def array_factor_single(self, element_complex_weights, u, v):
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        array_factor = 0 + 0j
        for element in range(self.num_elements):
            exponent = phases[element] + (
                1j * self.wavenumber * (element * self.spacing * u * v)
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
        array_factor = np.zeros((self.samples, self.samples), dtype=complex)
        for k in range(self.num_elements):
            element_angle = 2 * np.pi * k / self.num_elements
            exponent = phases[k] + 1j * self.wavenumber * (
                self.radius
                * (
                    np.sin(element_angle) * self.u_grid
                    + np.cos(element_angle) * self.v_grid
                )
            )
            array_factor += weights[k] * np.exp(exponent)
        array_factor = 20 * np.log10(np.abs(array_factor))
        return array_factor

    def array_factor_single(self, element_complex_weights, u, v):
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        array_factor = 0 + 0j
        for k in range(self.num_elements):
            element_angle = 2 * np.pi * k / self.num_elements
            exponent = phases[k] + 1j * self.wavenumber * (
                self.radius
                * (np.sin(element_angle) * u + np.cos(element_angle) * v)
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

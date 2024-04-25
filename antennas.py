"""Module containing class definitions of implemented antenna types"""

import bf_utils
import radiated_power_numba as rpn
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
        unit_circle = patches.Circle((0, 0), 1, color='k', fill=False)
        axis.clear()
        axis.set_xlabel("U")
        axis.set_ylabel("V")
        axis.imshow(array_factor, cmap='seismic', interpolation='nearest', extent=[-1, 1, -1, 1])
        axis.add_patch(unit_circle)

    def add_doi_indicators(self):
        """Adds marks on the array factor plots indicating the DOIs"""
        axes = [self.ax_tiled, self.ax_untiled]
        for target in self.targets:
            pos = bf_utils.spherical_to_uv(target)
            for axis in axes:
                axis.add_patch(patches.Circle((pos[0], pos[1]), 0.05, color='k', fill=False))

    def display(
        self,
        untiled_weights,
        tiled_weights,
        tile_labels,
        persist=False,
        pause_time=0.1,
    ):
        """Display the given untiled and tiled weights, including the selection pattern"""
        untiled_array_factor = 10 * np.log10(self.radiated_power(untiled_weights))
        self.update_array_factor_axis(self.ax_untiled, untiled_array_factor)
        self.ax_tiled.set_title("TILED: Array Factor")

        tiled_array_factor = 10 * np.log10(self.radiated_power(tiled_weights))
        self.update_array_factor_axis(self.ax_tiled, tiled_array_factor)
        self.ax_untiled.set_title("UNTILED: Array Factor")

        self.add_doi_indicators()
        self.update_tiling_plot(tile_labels)
        if persist:
            plt.show()
        else:
            plt.pause(pause_time)

    def estimate_MLE_region(self, doi, bw):
        """Estimate the region (by sample no.) required to calculate a DOI's MLE"""
        u_range = np.clip([doi[0] - bw, doi[0] + bw], -1, 1)
        v_range = np.clip([doi[1] - bw, doi[1] + bw], -1, 1)
        u_sample_range = np.rint(np.multiply(np.divide(np.add(u_range, 1), 2), (self.samples-1)))
        v_sample_range = np.rint(np.multiply(np.divide(np.add(v_range, 1), 2), (self.samples-1)))
        u_samples = np.arange(int(u_sample_range[0]), int(u_sample_range[1]), step=1, dtype=int)
        v_samples = np.arange(int(v_sample_range[0]), int(v_sample_range[1]), step=1, dtype=int)
        return u_samples, v_samples


        return np.asarray(u_sample_range, dtype=int), np.asarray(v_sample_range, dtype=int)

    def calculate_MLE(self, doi, beamwidth, radiated_power):
        """Estimate the lobe energy at the given DOI, with the defined complex weights"""
        u_sample_range, v_sample_range = self.estimate_MLE_region(doi, beamwidth)
        integration_constant = np.power(1/self.samples, 2)
        accumulator = 0
        for u_sample in u_sample_range:
            for v_sample in v_sample_range:
                u = self.u_grid[0][u_sample]
                v = self.v_grid[v_sample][0]
                radiated_power_sample = radiated_power[u_sample][v_sample]
                w = np.sqrt(1 - np.power(u, 2) - np.power(v, 2))
                accumulator += (radiated_power_sample / np.abs(w))
        mle = accumulator * integration_constant
        return mle

    def calculate_SLE(self, mle_sum, radiated_power):
        """Estimates SLE as the total energy minus the sum of all MLEs. Rough estimate."""
        integration_constant = np.power(1/self.samples, 2)
        w = 2 * np.pi / 3 # Estimate, w=2pi/3 when integrated over the unit disc, not correct but close enough.
        sle = (np.sum(radiated_power) * integration_constant / w) - mle_sum
        return sle

    def fitness(self, complex_weights, return_full_data=False):
        """Return the islr, and optionally other data, of a given set of complex weights
        mle - Main Lobe Energy
        sle - Sidelobe Energy
        islr - Integrated Sidelobe Ratio
        """
        radiated_power = self.radiated_power(complex_weights)
        mle_sum = 0
        mle = []
        for doi in self.targets:
            doi_uv = bf_utils.spherical_to_uv(doi[0:2])
            doi_bw = doi[2]
            doi_mle = self.calculate_MLE(doi_uv, doi_bw, radiated_power)
            mle_sum += doi_mle
            mle.append(doi_mle)
        sle = self.calculate_SLE(mle_sum, radiated_power)
        islr = sle/mle_sum
        if return_full_data:
            return islr, mle, mle_sum, sle
        return 1/islr

    def update_tiling_plot(self, tile_labels):
        """Update the tiling display subplot"""
        #pylint: disable=unused-argument
        assert False, "update_tiling_plot() must be defined in child class."

    def radiated_power(self, complex_weights):
        """Calculate the radiated_power for the given complex weights, returned in spherical coordinates (theta, phi)"""
        #pylint: disable=unused-argument
        assert False, "radiated_power() must be defined in child class."


class RectangularPlanar(Antenna):
    def __init__(self, frequency: float, spacing: tuple, num_el: tuple, parameters):
        num_elements = num_el[0] * num_el[1]
        self.num_el_x = num_el[0]
        self.num_el_y = num_el[1]
        self.spacing = spacing
        super().__init__(frequency, num_elements, parameters)

    def radiated_power(self, complex_weights):
        electric_field_preallocated = np.zeros((self.samples, self.samples), dtype=complex)
        return rpn.rpa_radiated_power(complex_weights, electric_field_preallocated, self.u_grid, self.v_grid, self.num_el_x, self.num_el_y, self.wavenumber, self.spacing)

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

    def radiated_power(self, complex_weights):
        electric_field_preallocated = np.zeros((self.samples, self.samples), dtype=complex)
        return rpn.ula_radiated_power(complex_weights, electric_field_preallocated, self.num_elements, self.wavenumber, self.spacing, self.u_grid)

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

    def radiated_power(self, complex_weights):
        electric_field_preallocated = np.zeros((self.samples, self.samples), dtype=complex)
        return rpn.ca_radiated_power(complex_weights, electric_field_preallocated, self.num_elements, self.radius, self.wavenumber, self.u_grid, self.v_grid)

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

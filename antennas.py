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
        wavelength = 3e8 / frequency
        self.wavenumber = 2 * np.pi / wavelength
        self.num_elements = num_elements
        self.samples = parameters.samples
        self.u_grid = parameters.u_grid
        self.v_grid = parameters.v_grid
        self.w_grid = parameters.w_grid
        self.figure = plt.figure()
        grid_spec = gridspec.GridSpec(2, 2)
        self.ax_tile_pattern = self.figure.add_subplot(grid_spec[:, 1])
        self.ax_untiled = self.figure.add_subplot(grid_spec[0, 0])
        self.ax_tiled = self.figure.add_subplot(grid_spec[1, 0])
        self.dois = parameters.dois
        self.dnois = parameters.dnois
        self.mse_target_levels = parameters.mse_target_levels
        self.mse_target = self.generate_mse_target()

    def generate_mse_target(self):
        mse_target = np.ones_like(self.u_grid) * self.mse_target_levels[1]
        for doi in self.dois:
            u_samples, v_samples = self.calculate_doi_region(doi)
            for u in u_samples:
                for v in v_samples:
                        mse_target[u, v] = self.mse_target_levels[0]
        for dnoi in self.dnois:
            u_samples, v_samples = self.calculate_doi_region(dnoi)
            for u in u_samples:
                for v in v_samples:
                        mse_target[u, v] = self.mse_target_levels[2]
        return mse_target


    def update_array_factor_axis(self, axis, array_factor):
        """Reset and clear axes for the next step's data output to be plotted"""
        unit_circle = patches.Circle((0, 0), 1, color="k", fill=False)
        axis.clear()
        axis.set_xlabel("U")
        axis.set_ylabel("V")
        axis.imshow(
            array_factor, cmap="seismic", interpolation="nearest", extent=[-1, 1, -1, 1]
        )
        axis.add_patch(unit_circle)

    def add_doi_indicators(self):
        """Adds marks on the array factor plots indicating the DOIs"""
        axes = [self.ax_tiled, self.ax_untiled]
        for doi in self.dois:
            pos = bf_utils.spherical_to_uv([doi[0], doi[1]])
            for axis in axes:
                axis.add_patch(
                    patches.Circle((pos[0], pos[1]), 0.1, color="g", fill=False)
                )
        for dnoi in self.dnois:
            pos = bf_utils.spherical_to_uv([dnoi[0], dnoi[1]])
            for axis in axes:
                axis.add_patch(
                    patches.Circle((pos[0], pos[1]), 0.1, color="k", fill=False)
                )

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

    def calculate_doi_region(self, doi):
        """Estimate the region (by sample no.) required to calculate a DOI's MLE"""
        u, v = bf_utils.spherical_to_uv([doi[0], doi[1]])
        bw = doi[2]
        u_range = np.clip([u - bw, u + bw], -1, 1)
        v_range = np.clip([v - bw, v + bw], -1, 1)
        u_sample_range = np.rint(
            np.multiply(np.divide(np.add(u_range, 1), 2), (self.samples))
        )
        v_sample_range = np.rint(
            np.multiply(np.divide(np.add(v_range, 1), 2), (self.samples))
        )
        u_samples = np.arange(
            int(u_sample_range[0]), int(u_sample_range[1]), step=1, dtype=int
        )
        v_samples = np.arange(
            int(v_sample_range[0]), int(v_sample_range[1]), step=1, dtype=int
        )
        return u_samples, v_samples

    def calculate_MLE(self, doi, radiated_power):
        """Estimate the lobe energy at the given DOI, with the defined complex weights"""
        u_sample_range, v_sample_range = self.calculate_doi_region(doi)
        integration_constant = np.power(2 / self.samples, 2)
        accumulator = 0
        for u_sample in u_sample_range:
            for v_sample in v_sample_range:
                u = self.u_grid[0][u_sample]
                v = self.v_grid[v_sample][0]
                w = self.w_grid[u_sample][v_sample]
                if np.abs(w) > 0:
                    accumulator += radiated_power[u_sample][v_sample] /np.abs(w)
        mle = accumulator * integration_constant
        return mle

    def calculate_SLE(self, mle_sum, radiated_power):
        """Estimates SLE as the total energy minus the sum of all MLEs. Rough estimate."""
        integration_constant = np.power(2 / self.samples, 2)
        accumulator = 0
        # Potential divide by zero here introduces NaNs, but the following line ignores those entries.
        sle_grid = np.divide(radiated_power, self.w_grid)
        sle = np.sum(sle_grid[np.abs(self.w_grid) > 0]) - mle_sum
        return sle

    def fitness(self, complex_weights):
        """Return the islr, and optionally other data, of a given set of complex weights
        mle - Main Lobe Energy
        sle - Sidelobe Energy
        islr - Integrated Sidelobe Ratio
        """
        radiated_power = self.radiated_power(complex_weights)
        doi_mle = []
        dnoi_mle = []
        for doi in self.dois:
            mle = self.calculate_MLE(doi, radiated_power)
            doi_mle.append(mle)
        for dnoi in self.dnois:
            mle = self.calculate_MLE(dnoi, radiated_power)
            dnoi_mle.append(mle)
        sle = self.calculate_SLE(np.sum(doi_mle), radiated_power)
        islr = sle / np.sum(doi_mle)
        array_factor = 10 * np.log10(radiated_power)
        normalised_af = array_factor - np.max(array_factor)
        mse = bf_utils.calculate_mean_squared_error(self.mse_target, normalised_af)
        islr = sle / np.sum(doi_mle)
        return mse, islr, doi_mle, dnoi_mle, sle


    def update_tiling_plot(self, tile_labels):
        """Update the tiling display subplot"""
        # pylint: disable=unused-argument
        assert False, "update_tiling_plot() must be defined in child class."

    def radiated_power(self, complex_weights):
        """Calculate the radiated_power for the given complex weights, returned in spherical coordinates (theta, phi)"""
        # pylint: disable=unused-argument
        assert False, "radiated_power() must be defined in child class."


class SimpleULA:
    def __init__(self, frequency, spacing, num_elements, parameters):
        self.frequency = frequency
        wavelength = 3e8 / frequency
        self.wavenumber = 2 * np.pi / wavelength
        self.num_elements = num_elements
        self.spacing = spacing
        self.samples = parameters.samples
        self.thetas = np.linspace(-np.pi/2, np.pi/2, self.samples)
        self.figure = plt.figure()
        grid_spec = gridspec.GridSpec(1, 1)
        self.ax_beam_pattern = self.figure.add_subplot(grid_spec[0, 0])
        self.dois = parameters.dois
        self.dnois = parameters.dnois
        self.mse_target_levels = parameters.mse_target_levels
        self.mse_target = self.generate_mse_target()

    def get_sample_range(self, angle, bw):
            value_range = np.clip([angle - bw, angle + bw], -np.pi/2, np.pi/2)
            sample_range = np.rint(
                np.multiply(np.divide(np.add(value_range, np.pi/2), np.pi), (self.samples - 1))
            )
            return np.asarray(sample_range, dtype=int)

    def generate_mse_target(self):
        mse_target_levels = self.mse_target_levels
        doi_af = mse_target_levels[0]
        default_af = mse_target_levels[1]
        dnoi_af = mse_target_levels[2]
        mse_target = np.asarray([default_af for _ in range(self.samples)])
        for doi in self.dois:
            theta_doi = doi[0]
            bw = doi[2]
            doi_sample_range = self.get_sample_range(theta_doi, bw)
            mse_target[doi_sample_range[0]:doi_sample_range[1]] = doi_af
        for dnoi in self.dnois:
            dnoi_theta = dnoi[0]
            bw = dnoi[2]
            dnoi_sample_range = self.get_sample_range(dnoi_theta, bw)
            mse_target[dnoi_sample_range[0]:dnoi_sample_range[1]] = dnoi_af
        return mse_target

    def update_array_factor_axis(self, axis, array_factor):
        """Reset and clear axes for the next step's data output to be plotted"""
        axis.clear()
        axis.set_xlabel("Theta (degrees)")
        axis.set_ylim(self.mse_target_levels[2]-10, 10)
        axis.set_ylabel("Normalised Array Factor (dB)")
        axis.plot(array_factor)
        axis.plot(self.mse_target, 'red')
        axis.set_xticks(np.linspace(0, self.samples, 5), labels=np.linspace(-90, 90, 5, dtype=int))

    def display(
        self,
        complex_weights,
        _1,
        _2,
        persist=False,
        pause_time=0.25,
    ):
        """Display the given untiled and tiled weights, including the selection pattern"""
        # Really nasty function signature here but that's ok. :)
        array_factor = 10 * np.log10(self.radiated_power(complex_weights))
        normalised_af = array_factor - np.max(array_factor)
        self.update_array_factor_axis(self.ax_beam_pattern, normalised_af)
        self.ax_beam_pattern.set_title("Array Factor")
        if persist:
            plt.show()
        else:
            plt.pause(pause_time)

    def fitness(self, complex_weights):
        """Return the islr, and optionally other data, of a given set of complex weights
        mle - Main Lobe Energy
        sle - Sidelobe Energy
        islr - Integrated Sidelobe Ratio
        mse - Mean Squared Error
        """
        radiated_power = self.radiated_power(complex_weights)
        array_factor = 10 * np.log10(radiated_power)
        normalised_af = array_factor - np.max(array_factor)
        mse = bf_utils.calculate_mean_squared_error(self.mse_target, normalised_af)
        doi_mle = []
        dnoi_mle = []
        doi_af = []
        dnoi_af = []
        for doi in self.dois:
            theta = doi[0]
            bw = doi[2]
            mle = self.calculate_MLE(theta, bw, radiated_power)
            doi_mle.append(mle)
            theta_sample = np.multiply(np.divide(np.add(theta, np.pi/2), np.pi), (self.samples))
            doi_af.append(array_factor[int(theta_sample)])
        for dnoi in self.dnois:
            theta = dnoi[0]
            bw = dnoi[2]
            mle = self.calculate_MLE(theta, bw, radiated_power)
            dnoi_mle.append(mle)
            theta_sample = np.multiply(np.divide(np.add(theta, np.pi/2), np.pi), (self.samples))
            dnoi_af.append(array_factor[int(theta_sample)])
        sle = self.calculate_SLE(np.sum(doi_mle), radiated_power)
        islr = sle / np.sum(doi_mle)
        af_difference = np.sum(doi_af) - np.sum(dnoi_af)
        return mse, islr, doi_mle, dnoi_mle, sle, af_difference


    def calculate_MLE(self, theta, bw, radiated_power):
        theta_range = np.clip([theta - bw, theta + bw], -np.pi/2, np.pi/2)
        theta_sample_range = np.rint(
            np.multiply(np.divide(np.add(theta_range, np.pi/2), np.pi), (self.samples))
        )
        integration_region = radiated_power[int(theta_sample_range[0]):int(theta_sample_range[1])]
        mle = np.sum(integration_region) * 1/self.samples
        return mle

    def calculate_SLE(self, mle_sum, radiated_power):
        sle = (np.sum(radiated_power) * 1/self.samples) - mle_sum
        return sle

    def radiated_power(self, complex_weights):
        """Calculate the radiated_power for the given complex weights"""
        electric_field = np.zeros_like(self.thetas, dtype=complex)
        phases = np.angle(complex_weights)
        weights = np.abs(complex_weights)
        sin_thetas = np.sin(self.thetas)
        for element in range(self.num_elements):
            exponent = 1j * (phases[element] + (self.wavenumber * element * self.spacing * sin_thetas))
            electric_field += weights[element] * np.exp(exponent)
        radiated_power = np.power(np.abs(electric_field), 2)
        return radiated_power


class RectangularPlanar(Antenna):
    def __init__(self, frequency: float, spacing: tuple, num_el: tuple, parameters):
        num_elements = num_el[0] * num_el[1]
        self.num_el_x = num_el[0]
        self.num_el_y = num_el[1]
        self.spacing = spacing
        super().__init__(frequency, num_elements, parameters)

    def radiated_power(self, complex_weights):
        electric_field_preallocated = np.zeros(
            (self.samples, self.samples), dtype=complex
        )
        return rpn.rpa_radiated_power(
            complex_weights,
            electric_field_preallocated,
            self.u_grid,
            self.v_grid,
            self.num_el_x,
            self.num_el_y,
            self.wavenumber,
            self.spacing,
        )

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
    # use SimpleULA instead.
    def __init__(self, frequency: float, spacing: float, num_el: int, parameters):
        self.spacing = spacing
        super().__init__(frequency, num_el, parameters)

    def radiated_power(self, complex_weights):
        electric_field_preallocated = np.zeros(
            (self.samples, self.samples), dtype=complex
        )
        return rpn.ula_radiated_power(
            complex_weights,
            electric_field_preallocated,
            self.num_elements,
            self.wavenumber,
            self.spacing,
            self.u_grid,
        )

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
    #Not fully implemented, was used for earlier testing.
    def __init__(self, frequency, radius, num_elements, parameters):
        self.radius = radius
        super().__init__(frequency, num_elements, parameters)

    def radiated_power(self, complex_weights):
        electric_field_preallocated = np.zeros(
            (self.samples, self.samples), dtype=complex
        )
        return rpn.ca_radiated_power(
            complex_weights,
            electric_field_preallocated,
            self.num_elements,
            self.radius,
            self.wavenumber,
            self.u_grid,
            self.v_grid,
        )

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

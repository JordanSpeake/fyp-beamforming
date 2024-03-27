"""Classes for defining antennas, in most cases calculations are taken from:
    Digital Beamforming in Wireless Communications, J. Litva, T. Kwok-Yeung Lo
    unless otherwise specified."""

import numpy as np
import matplotlib.pyplot as plt


class UniformLinear:
    """A uniform linear array antenna, centered on 0,0,0"""

    def __init__(self, frequency, spacing, num_elements):
        self.frequency = frequency
        self.spacing = spacing
        self.num_elements = num_elements
        N = (num_elements - 1) / 2
        self.wavelength = 3e9 / frequency
        self.wavenumber = 2 * np.pi / self.wavelength
        self.dimensions = 1
        self.positions = np.linspace(self.spacing * -N, self.spacing * N, num_elements)
        # self.x_coords, self.y_coords = np.meshgrid(x, y)
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(projection="3d")

    def fitness(self, element_complex_weights, parameters):
        """Calculate the fitness of a given set of antenna element weights and phases
        Takes a particle's position as the antenna's complex weights"""
        array_factor = self.array_factor(element_complex_weights, parameters)
        beamwidth = parameters.beamwidth_samples
        score = 0
        for target in parameters.targets:
            beam_range = np.arange(target - beamwidth, target + beamwidth)
            score += np.average(array_factor[beam_range])
        return score

    def array_factor(self, element_complex_weights, parameters):
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        theta = np.tile(np.atleast_2d(parameters.theta), (parameters.samples, 1))
        phi = np.tile(np.atleast_2d(parameters.phi).T, (1, parameters.samples))
        sin_u = np.sin(theta) * np.cos(phi)
        array_factor = np.zeros((parameters.samples, parameters.samples), dtype=complex)
        for m in range(self.num_elements):
            exponent = phases[m] + (
                self.wavenumber * m * self.spacing * np.sin(theta) * np.cos(phi)
            )
            array_factor += weights[m] * np.exp(1j * exponent)
        array_factor = 20 * np.log10(np.abs(array_factor))
        return array_factor

    def display(self, element_complex_weights, parameters, persist):
        array_factor = self.array_factor(element_complex_weights, parameters)
        R, P = np.meshgrid(parameters.phi, parameters.theta)
        X, Y = R * np.cos(P), R * np.sin(P)
        self.axes.plot_surface(X, Y, array_factor, cmap=plt.cm.YlGnBu_r)
        if persist:
            plt.show()
        else:
            plt.pause(0.05)


class Circular:
    def __init__(self, frequency, radius, num_elements):
        self.radius = radius
        self.num_elements = num_elements
        self.frequency = frequency
        self.wavelength = 3e9 / frequency
        self.element_angles = np.linspace(
            0, 2 * np.pi - (2 * np.pi / num_elements), num_elements
        )
        self.wavenumber = 2 * np.pi / self.wavelength
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(projection="3d")

    def fitness(self, element_complex_weights, parameters):
        array_factor = self.array_factor(element_complex_weights, parameters)
        score = 0
        for target in parameters.targets:
            score += array_factor[target[0]][target[1]]
        return score

    def array_factor(self, element_complex_weights, parameters):
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        array_factor = np.zeros((parameters.samples, parameters.samples), dtype=complex)
        for k in range(self.num_elements):
            sin_thetas = np.atleast_2d(np.sin(parameters.theta))
            cos_phi_deltas = np.atleast_2d(
                np.cos(parameters.phi - self.element_angles[k])
            ).T
            sin_cos_product = np.matmul(cos_phi_deltas, sin_thetas)
            exponent = phases[k] - self.wavenumber * self.radius * sin_cos_product
            array_factor += weights[k] * np.exp(1j * exponent)
        array_factor = 20 * np.log10(np.abs(array_factor))
        return array_factor

    def display(self, element_complex_weights, parameters, persist):
        array_factor = self.array_factor(element_complex_weights, parameters)
        R, P = np.meshgrid(parameters.phi, parameters.theta)
        X, Y = R * np.cos(P), R * np.sin(P)
        self.axes.plot_surface(X, Y, array_factor, cmap=plt.cm.YlGnBu_r)
        if persist:
            plt.show()
        else:
            plt.pause(0.05)


class RectangularPlanar:
    def __init__(self, frequency, spacing, num_elements):
        """Definition for a RPA. 'spacing' and 'num_elements' must be given in tuple form (x, y)"""
        self.frequency = frequency
        self.spacing: tuple = spacing
        self.spacing_x: float = spacing[0]
        self.spacing_y: float = spacing[1]
        self.num_elements: tuple = num_elements
        self.num_elements_x: int = num_elements[0]
        self.num_elements_y: int = num_elements[1]
        self.wavelength = 3e9 / frequency
        self.wavenumber = 2 * np.pi / self.wavelength
        x = np.linspace(0, 1, self.num_elements_x)
        y = np.linspace(0, 1, self.num_elements_y)
        self.x_coords, self.y_coords = np.meshgrid(x, y)
        self.figure = plt.figure()
        self.axes = self.figure.add_subplot(projection="3d")

    def fitness(self, element_complex_weights, parameters):
        array_factor = self.array_factor(element_complex_weights, parameters)
        score = 0
        for target in parameters.targets:
            score += array_factor[target[0]][target[1]]
        return score

    def display(self, element_complex_weights, parameters, persist):
        array_factor = self.array_factor(element_complex_weights, parameters)
        R, P = np.meshgrid(parameters.phi, parameters.theta)
        X, Y = R * np.cos(P), R * np.sin(P)
        self.axes.plot_surface(X, Y, array_factor, cmap=plt.cm.YlGnBu_r)
        if persist:
            plt.show()
        else:
            plt.pause(0.05)

    def array_factor(self, element_complex_weights, parameters):
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        array_factor = np.zeros((parameters.samples, parameters.samples), dtype=complex)

        theta = np.tile(np.atleast_2d(parameters.theta), (parameters.samples, 1))
        phi = np.tile(np.atleast_2d(parameters.phi).T, (1, parameters.samples))
        sin_u = np.sin(theta) * np.cos(phi)
        sin_v = np.sin(theta) * np.sin(phi)
        array_factor = np.zeros((parameters.samples, parameters.samples), dtype=complex)
        for m in range(self.num_elements_x):
            for n in range(self.num_elements_y):
                phase = phases[m][n]
                weight = weights[m][n]
                exponent = (
                    self.wavenumber
                    * (m * self.spacing_x * sin_u + n * self.spacing_y * sin_v)
                    + phase
                )
                array_factor += weight * np.exp(1j * exponent)
        array_factor = 20 * np.log10(np.abs(array_factor))
        return array_factor

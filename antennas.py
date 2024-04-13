import numpy as np
import matplotlib.pyplot as plt


class Antenna:
    def __init__(self, frequency, num_elements, parameters):
        self.frequency = frequency
        self.wavelength = 3e9 / frequency
        self.wavenumber = 2 * np.pi / self.wavelength
        self.theta = np.tile(np.atleast_2d(parameters.theta), (parameters.samples, 1))
        self.phi = np.tile(np.atleast_2d(parameters.phi).T, (1, parameters.samples))
        self.theta_samples = parameters.theta
        self.phi_samples = parameters.phi
        self.sin_u = np.sin(self.theta) * np.cos(self.phi)
        self.sin_v = np.sin(self.theta) * np.sin(self.phi)
        self.num_elements = num_elements
        self.figure, (self.ax_untiled, self.ax_tiled) = plt.subplots(1, 2, subplot_kw={'projection': "3d"})
        self.setup_figures()

    def setup_figures(self):
        self.ax_tiled.set_title("ORIGINAL: Array Factor")
        self.ax_tiled.set_xlabel("Angle")
        self.ax_tiled.set_ylabel("Array Factor (dB)")
        self.ax_untiled.set_title("TILED: Array Factor")
        self.ax_untiled.set_xlabel("Angle")
        self.ax_untiled.set_ylabel("Array Factor (dB)")

    def display(self, untiled_weights, tiled_weights, persist=False, pause_time=0.01):
        untiled_af = self.array_factor(untiled_weights)
        tiled_af = self.array_factor(tiled_weights)
        R, P = np.meshgrid(self.phi_samples, self.theta_samples)
        X, Y = R * np.cos(P), R * np.sin(P)
        self.ax_untiled.plot_surface(X, Y, untiled_af, cmap=plt.cm.YlGnBu_r)
        self.ax_tiled.plot_surface(X, Y, tiled_af, cmap=plt.cm.YlGnBu_r)
        if persist:
            plt.show()
        else:
            plt.pause(pause_time)

    def fitness(self, element_complex_weights, parameters):
        #TODO replace with LMS
        score = 0
        for target in parameters.targets:
            score += self.array_factor_single(
                element_complex_weights, target[0], target[1]
            )
        return score

    def array_factor_single(self, element_complex_weights, theta, phi):
        assert False, "array_factor_single() must be defined in child class."

    def array_factor(self, element_complex_weights):
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

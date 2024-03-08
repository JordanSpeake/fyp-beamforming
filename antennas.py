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
        """Calculate the array factor of a given set of antenna element weights and phases"""
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        theta = parameters.theta
        array_factor = np.zeros(parameters.samples, dtype=complex)
        for i, theta_val in enumerate(theta):
            exponent = self.wavenumber * np.sin(theta_val) * self.positions + phases
            array_factor[i] = np.sum(weights * np.exp(1j * exponent))
        array_factor = 20 * np.log10(np.abs(array_factor))
        return array_factor

    def display(self, element_complex_weights, parameters, persist):
        array_factor = antenna.array_factor(position, parameters)
        plt.plot(parameters.theta, array_factor)

        # targets_markers = (2 * np.pi * parameters.targets / parameters.samples) - np.pi
        # for target in targets_markers:
        #     plt.axvspan(
        #         target - parameters.beamwidth,
        #         target + parameters.beamwidth,
        #         color="green",
        #         alpha=0.5,
        #     )

        # peaks, _ = find_peaks(array_factor, height=-50, distance=5)
        # peak_angles = (2 * np.pi * peaks / parameters.samples) - np.pi
        # plt.plot(peak_angles, array_factor[peaks], "X", color="orange")

        plt.xlim(-np.pi / 2, np.pi / 2)
        plt.ylim((-40, 0))
        plt.xlabel("Beam angle [rad]")
        plt.ylabel("Power [dB]")
        plt.clf()
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
        self.dimensions = 2
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
        theta = parameters.theta
        phi = parameters.phi
        array_factor = np.zeros((parameters.samples, parameters.samples), dtype=complex)
        for i in range(len(phi)):
            for j in range(len(theta)):
                exponent = phases - (
                    self.wavenumber
                    * self.radius
                    * np.cos(phi[i] - self.element_angles)
                    * np.sin(theta[j])
                )
                array_factor[i][j] = np.sum(weights * np.exp(1j * exponent))
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

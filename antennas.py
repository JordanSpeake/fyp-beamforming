import numpy as np


class ULA:
    def __init__(self, frequency, spacing_coeff, num_elements):
        self.num_elements = num_elements
        self.frequency = frequency
        self.wavelength = 3e9 / frequency
        self.spacing = self.wavelength * spacing_coeff
        N = (num_elements - 1) / 2
        self.positions = np.linspace(self.spacing * -N, self.spacing * N, num_elements)
        self.wavenumber = 2 * np.pi / self.wavelength

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
        array_factor = 20 * np.log10(
            np.abs(self.far_zone_e_r(element_complex_weights, parameters))
        )
        return array_factor  - np.max(array_factor)

    def far_zone_e_r(self, element_complex_weights, parameters):
        """Calculate the far-zone electric field of a given set of antenna element weights and phases"""
        # TODO - hot code
        e_r = np.zeros(parameters.samples, dtype=np.csingle)
        # offset because of calculation funnybusiness... should refactor function
        phi = parameters.phi + np.pi / 2
        phases = np.angle(element_complex_weights)
        weights = np.abs(element_complex_weights)
        for i in range(parameters.samples):
            angle = phi[i]
            theta = self.wavenumber * np.cos(angle) * self.positions + phases
            e_t = np.sum(weights * np.exp(1j * theta))
            e_r[i] = e_t / self.num_elements
        return e_r

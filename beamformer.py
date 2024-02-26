import numpy as np

def define_ULA(frequency, spacing_coeff, num_elements):
    wavelength = 3e9/frequency
    spacing = wavelength * spacing_coeff
    N = (num_elements - 1)/2
    positions = np.linspace(spacing*-N, spacing*N, num_elements)
    return {
        "elements" : num_elements,
        "spacing" : spacing,
        "positions" : positions,
        "frequency" : frequency,
        "wavelength" : wavelength,
        "wavenumber" : 2*np.pi / wavelength,
    }

def define_parameters(samples, pop_size):
    return {
        "samples" : samples,
        "population_size" : pop_size,
    }

def beamformer(antenna, parameters):
    feedback_antenna = f"Antenna: {antenna}"
    feedback_parameters = f"Parameters: {parameters}"
    return f"{feedback_antenna} \n {feedback_parameters}"

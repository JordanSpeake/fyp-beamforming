import numpy as np
from matplotlib import pyplot as plt
from scipy import constants, signal
import bf_utils

def radiated_power_linear_shift(weights, angle, num_elements, samples, wavenumber, spacing):
    electric_field = np.zeros(samples, dtype=complex)
    sin_thetas = np.sin(np.linspace(-np.pi/2, np.pi/2, samples))
    for element in range(num_elements):
        phase = wavenumber * spacing * -np.sin(angle)
        exponent = 1j * ((element * phase) + (wavenumber * element * spacing * sin_thetas))
        electric_field += weights[element] * np.exp(exponent)
    radiated_power = np.power(np.abs(electric_field), 2)
    return radiated_power

def calculate_MLE(radiated_power, theta, bw, samples):
    theta_range = np.clip([theta - bw, theta + bw], -np.pi/2, np.pi/2)
    theta_sample_range = np.rint(
        np.multiply(np.divide(np.add(theta_range, np.pi/2), np.pi), (samples))
    )
    integration_region = radiated_power[int(theta_sample_range[0]):int(theta_sample_range[1])]
    mle = np.sum(integration_region) * 1/samples
    return mle

def calculate_SLE(radiated_power, theta, bw, mle_sum, samples):
    sle = (np.sum(radiated_power) * 1/samples) - mle_sum
    return sle

def calculate_ISLR(radiated_power, target_angle, beamwidth, samples):
    mle = calculate_MLE(radiated_power, target_angle, beamwidth, samples)

    sle = calculate_SLE(radiated_power, target_angle, beamwidth, mle, samples)
    islr = mle/sle
    return islr, mle

def beamformer(weights, angle, num_elements, samples, wavenumber, spacing, beamwidth):
    rad_power = radiated_power_linear_shift(weights, angle, num_elements, samples, wavenumber, spacing)
    array_factor = 10 * np.log10(rad_power)
    normalised_af = array_factor - np.max(array_factor)
    islr, mle = calculate_ISLR(rad_power, angle, beamwidth, samples)
    print(f"    Angle: {angle}")
    print(f"        ISLR: {islr}, {bf_utils.to_dB(islr)}")
    print(f"        MLE: {mle}, {bf_utils.to_dB(mle)}")
    plt.plot(normalised_af)
    plt.show()


def main():
    num_elements = 8
    weights = np.linspace(1, 1, num_elements)
    samples = 300
    frequency = 30e8
    wavelength = 3e8 / frequency
    wavenumber = 2 * np.pi / wavelength
    spacing = wavelength/2
    beamformer(weights, np.deg2rad(60), num_elements, samples, wavenumber, spacing, 10)
    # for beamwidth in [10, 20, 30]:
    #     print(f"BEAMWIDTH: {beamwidth}")
    #     beamwidth = np.deg2rad(beamwidth)
    #     for angle in [0, 15, 30, 45, 60]:
    #         angle = np.deg2rad(angle)
    #         beamformer(weights, angle, num_elements, samples, wavenumber, spacing, beamwidth)

main()

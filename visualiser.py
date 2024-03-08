import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences


def display(position, antenna, parameters, persist=False):
    if antenna.dimensions == 1:
        display1D(position, antenna, parameters, persist)
    elif antenna.dimensions == 2:
        display2D(position, antenna, parameters, persist)
    else:
        print("Cannot display array factor, bad dimensions.")


def display2D(position, antenna, parameters, persist):
    array_factor = antenna.array_factor(position, parameters)
    plt.subplot(2, 1, 1)
    plt.imshow(array_factor, interpolation="bilinear")
    # X, Y = np.meshgrid(parameters.theta, parameters.phi)
    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # ax.plot_surface(X, Y, array_factor)
    # plt.xlim(-np.pi / 2, np.pi / 2)
    # plt.ylim((-40, 0))
    # plt.xlabel("Beam angle [rad]")
    # plt.ylabel("Power [dB]")
    plt.show()


def display1D(position, antenna, parameters, persist):
    array_factor = antenna.array_factor(position, parameters)
    plt.clf()
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

    # plt.xlim(-np.pi / 2, np.pi / 2)
    # plt.ylim((-40, 0))
    # plt.xlabel("Beam angle [rad]")
    # plt.ylabel("Power [dB]")
    if persist:
        plt.show()
    else:
        plt.pause(0.05)

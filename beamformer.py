import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def calculate_array_factor(position, antenna, parameters):
    array_factor = 20 * np.log10(np.abs(calculate_er(position, antenna, parameters)))
    return array_factor - np.max(array_factor)


def calculate_er(position, antenna, parameters):
    # TODO - hot code
    e_r = np.zeros(parameters["samples"], dtype=np.csingle)
    # offset because of calculation funnybusiness... should refactor function
    phi = parameters["phi"] + np.pi / 2
    phases, weights = np.split(position, 2)
    for i in range(parameters["samples"]):
        angle = phi[i]
        theta = antenna["wavenumber"] * np.cos(angle) * antenna["positions"] + phases
        e_t = np.sum(weights * np.exp(1j * theta))
        e_r[i] = e_t / antenna["num_elements"]
    return e_r


def new_population(antenna, parameters):
    population = []
    for i in range(parameters["population_size"]):
        population.append(new_particle(antenna, parameters))
    return population


def calculate_sidelobe_level(array_factor, targets):
    # TODO make this more accurate, average will do for now
    peaks, properties = find_peaks(array_factor, height=-50, distance=5)
    heights = np.sort(properties["peak_heights"])
    sidelobe_level = np.average(heights)
    return sidelobe_level, peaks, properties


def fitness(position, antenna, parameters):
    array_factor = calculate_array_factor(position, antenna, parameters)
    beamwidth = parameters["beamwidth_samples"]
    score = 0
    for target in parameters["targets"]:
        beam_range = np.arange(target - beamwidth, target + beamwidth)
        score += np.sum(array_factor[beam_range]) * 2
        array_factor = np.delete(array_factor, beam_range)
    sidelobe_level, _, _ = calculate_sidelobe_level(array_factor, parameters["targets"])
    sidelobe_penalty = np.exp(
        -parameters["sidelobe_suppression"] / (sidelobe_level - 0.01)
    )
    score -= sidelobe_penalty
    return score


def new_particle(antenna, parameters):
    phases = np.random.rand(antenna["num_elements"]) * 2 * np.pi
    weights = np.random.rand(antenna["num_elements"])
    position = np.concatenate((phases, weights))
    score = fitness(position, antenna, parameters)
    return {
        "best_position": position,
        "position": position,
        "velocity": np.random.rand(2 * antenna["num_elements"]),
        "score": score,
    }


def display(position, antenna, parameters, persist=False):
    array_factor = calculate_array_factor(position, antenna, parameters)
    plt.clf()
    plt.plot(parameters["phi"], array_factor)

    targets_markers = (
        2 * np.pi * parameters["targets"] / parameters["samples"]
    ) - np.pi
    for target in targets_markers:
        plt.axvspan(
            target - parameters["beamwidth"],
            target + parameters["beamwidth"],
            color="green",
            alpha=0.5,
        )

    sidelobe_level, peaks, _ = calculate_sidelobe_level(
        array_factor, parameters["targets"]
    )
    peak_angles = (2 * np.pi * peaks / parameters["samples"]) - np.pi
    plt.plot(peak_angles, array_factor[peaks], "X", color="orange")

    print(f"sidelobe: {sidelobe_level}")
    plt.axhline(sidelobe_level, color="orange", alpha=0.5)
    plt.xlim(-np.pi / 2, np.pi / 2)
    plt.ylim((-50, 0))
    plt.xlabel("Beam angle [rad]")
    plt.ylabel("Power [dB]")
    if persist:
        plt.show()
    else:
        plt.pause(0.05)



def clip_position(particle_position):
    """Clip phase position to [0, 2pi] and weights to [0, 1]"""
    phases, weights = np.split(particle_position, 2)
    return np.concatenate((np.clip(phases, 0, 2 * np.pi), np.clip(weights, 0, 1)))


def clip_velocity(particle_velocity, parameters):
    """Clip velocity to +/- parameters["max_particle_velocity"]"""
    return np.clip(
        particle_velocity,
        -parameters["max_particle_velocity"],
        parameters["max_particle_velocity"],
    )


def update_particle(particle, best_known_position, antenna, parameters):
    inertial_component = parameters["intertia_weight"] * particle["velocity"]
    cognitive_component = (
        parameters["cognitive_coeff"]
        * np.random.rand(2 * antenna["num_elements"])
        * (np.subtract(particle["best_position"], particle["position"]))
    )
    social_component = (
        parameters["social_coeff"]
        * np.random.rand(2 * antenna["num_elements"])
        * (np.subtract(best_known_position, particle["position"]))
    )
    particle["velocity"] = clip_velocity(
        inertial_component + cognitive_component + social_component, parameters
    )
    particle["position"] = clip_position(
        np.add(particle["position"], particle["velocity"])
    )
    return particle


def step_PSO(population, best_known_position, best_score, antenna, parameters):
    for particle in population:
        particle = update_particle(particle, best_known_position, antenna, parameters)
        score = fitness(particle["position"], antenna, parameters)
        if score > best_score:
            best_known_position = particle["position"]
            best_score = score
        if score > particle["score"]:
            particle["best_position"] = particle["position"]
            particle["score"] = score
    return population, best_known_position, best_score


def particle_swarm_optimisation(antenna, parameters, logging):
    population = new_population(antenna, parameters)
    best_known_position = np.random.rand(2 * antenna["num_elements"]) * 2 * np.pi
    best_score = fitness(best_known_position, antenna, parameters)
    for step in range(0, parameters["max_steps"]):
        population, best_known_position, best_score = step_PSO(
            population, best_known_position, best_score, antenna, parameters
        )
        step += 1
        if logging["verbose"]:
            print(f"Step: {step}")
            print(f"Position: {best_known_position}\n Score: {best_score}")
        if logging["show_plots"]:
            display(best_known_position, antenna, parameters)
    return best_known_position


def beamformer(antenna, parameters, logging):
    result = particle_swarm_optimisation(antenna, parameters, logging)
    if logging["show_plots"] and logging["plots_persist"]:
        display(result, antenna, parameters, persist=True)
    return result

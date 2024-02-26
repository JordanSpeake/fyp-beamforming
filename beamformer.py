import numpy as np
import matplotlib.pyplot as plt

def calculate_array_factor(phases, antenna, parameters):
    array_factor = 20 * np.log10(np.abs(calculate_er(phases, antenna, parameters)))
    return array_factor

def calculate_er(phases, antenna, parameters):
    # TODO - hot code
    e_r = np.zeros(parameters["samples"], dtype=np.csingle)
    phi = parameters["phi"]
    for i in range(parameters["samples"]):
        angle = phi[i]
        theta = antenna["wavenumber"] * np.cos(angle) * antenna["positions"] + phases
        e_t = np.sum(antenna["weights"] * np.exp(1j * theta))
        e_r[i] = e_t/antenna["num_elements"]
    return e_r

def new_population(antenna, parameters):
    population = []
    for i in range(parameters["population_size"]):
        population.append(new_particle(antenna, parameters))
    return population

def fitness(position, antenna, parameters):
    array_factor = calculate_array_factor(position, antenna, parameters)
    score = 0
    for target in parameters["target_samples"]:
        score += array_factor[target]
        # score += (array_factor[target-1] + array_factor[target] + array_factor[target+1])/3
    return score

def new_particle(antenna, parameters):
    position = np.random.rand(antenna["num_elements"]) *2*np.pi
    score = fitness(position, antenna, parameters)
    return {
        "best_position" : position,
        "position" : position,
        "velocity" : np.random.rand(antenna["num_elements"]),
        "score" : score
    }

def display(position, antenna, parameters):
    array_factor = calculate_array_factor(position, antenna, parameters)
    plt.clf()
    plt.plot(parameters["phi"]-np.pi/2, array_factor)
    for target in parameters["target_samples"]:
        plt.axvline(2*target/parameters["samples"], color="red")
    plt.xlim(-np.pi/2, np.pi/2)
    plt.ylim((-50, 0))
    plt.xlabel('Beam angle [rad]')
    plt.ylabel('Power [dB]')
    plt.pause(0.01)

def define_parameters(pop_size, angular_samples, cognitive_coeff, social_coeff, intertia_weight, max_steps, static_targets):
    phi = np.linspace(0, np.pi, angular_samples)
    target_samples = np.asarray(static_targets) * angular_samples / np.pi
    target_samples = target_samples.astype(int)
    return {
        "population_size" : pop_size,
        "samples" : angular_samples,
        "phi" : phi,
        "cognitive_coeff" : cognitive_coeff,
        "social_coeff" : social_coeff,
        "intertia_weight" : intertia_weight,
        "max_steps" : max_steps,
        "target_samples" : target_samples,
    }

def define_ULA(frequency, spacing_coeff, num_elements):
    wavelength = 3e9/frequency
    spacing = wavelength * spacing_coeff
    N = (num_elements - 1)/2
    positions = np.linspace(spacing*-N, spacing*N, num_elements)
    return {
        "num_elements" : num_elements,
        "spacing" : spacing,
        "positions" : positions,
        "frequency" : frequency,
        "wavelength" : wavelength,
        "wavenumber" : 2*np.pi / wavelength,
        "weights" : np.ones(num_elements)
    }

def particle_swarm_optimisation(antenna, parameters):
    population = new_population(antenna, parameters)
    best_known_position = np.random.rand(antenna["num_elements"]) *2*np.pi
    best_score = -1000
    step = 0
    while step <= parameters["max_steps"]:
        print(f"Step: {step}")
        population, best_known_position, best_score = step_PSO(population, best_known_position, best_score, antenna, parameters)
        step += 1
        print(f"Position: {best_known_position}\n Score: {best_score}")
        display(best_known_position, antenna, parameters)
    return best_known_position

def step_PSO(population, best_known_position, best_score, antenna, parameters):
    for particle in population:
        a = parameters["intertia_weight"]*particle["velocity"]
        b = parameters["cognitive_coeff"]*np.random.rand(antenna["num_elements"])*(np.subtract(particle["best_position"], particle["position"]))
        c = parameters["social_coeff"]*np.random.rand(antenna["num_elements"])*(np.subtract(best_known_position, particle["position"]))
        particle["velocity"] = a + b + c # TODO refactor readability
        particle["position"] = np.add(particle["position"], particle["velocity"])
        score = fitness(particle["position"], antenna, parameters)
        if score > best_score:
            best_known_position = particle["position"]
            best_score = score
            particle["best_position"] = particle["position"]
            particle["score"] = score
        elif score > particle["score"]:
            particle["best_position"] = particle["position"]
            particle["score"] = score
    return population, best_known_position, best_score

def beamformer(antenna, parameters):
    best_found_position = particle_swarm_optimisation(antenna, parameters)
    display(best_found_position, antenna, parameters)
    return result

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences


class Particle:
    def __init__(self, antenna, parameters):
        self.velocity = np.random.uniform(antenna.num_elements) * np.exp(
            1j * np.random.uniform(antenna.num_elements) * 2 * np.pi
        )
        self.position = np.random.uniform(antenna.num_elements) * np.exp(
            1j * np.random.uniform(antenna.num_elements) * 2 * np.pi
        )
        self.best_position = self.position
        self.best_known_position = self.position
        self.score = fitness(self.position, antenna, parameters)
        self.best_score = self.score
        self.best_known_score = self.score
        self.dimensions = antenna.num_elements
        self.max_velocity = parameters.max_particle_velocity
        self.inertia_weight = parameters.intertia_weight
        self.cognitive_coeff = parameters.cognitive_coeff
        self.social_coeff = parameters.social_coeff

    def step(self, antenna, parameters):
    # TODO should the random numbers here be complex as well? or just reals?
        inertial_component = self.inertia_weight * self.velocity
        cognitive_component = (
            self.cognitive_coeff
            * np.random.rand(self.dimensions)
            * (np.subtract(self.best_position, self.position))
        )
        social_component = (
            self.social_coeff
            * np.random.rand(self.dimensions)
            * (np.subtract(self.best_known_position, self.position))
        )
        self.set_velocity(inertial_component + cognitive_component + social_component)
        self.set_position(np.add(self.position, self.velocity))
        self.update_score(antenna, parameters)

    def update_score(self, antenna, parameters):
        self.score = fitness(self.position, antenna, parameters)
        if self.score > self.best_score:
            self.best_position = self.position
            self.best_score = self.score
        if self.score > self.best_known_score:
            self.best_known_position = self.position
            self.best_known_score = self.score


    def set_position(self, position):
        """Set and limit magnitude (weights) of particle to 1 and wrap its angle (phase) [0, 2pi]"""
        weights = np.clip(np.abs(position), 0, 1)
        phases = np.mod(np.angle(position), 2 * np.pi)
        self.position = weights * np.exp(1j * phases)

    def set_velocity(self, velocity):
        """Set and limit velocity of particle to max_particle_velocity and wrap direction [0, 2pi]"""
        speed = np.clip(np.abs(velocity), 0, self.max_velocity)
        direction = np.mod(np.angle(velocity), 2 * np.pi)
        self.velocity =  speed * np.exp(1j * direction)



def new_population(antenna, parameters):
    population = []
    for i in range(parameters.population_size):
        population.append(Particle(antenna, parameters))
    return population

def calculate_array_factor(position, antenna, parameters):
    array_factor = 20 * np.log10(np.abs(calculate_er(position, antenna, parameters)))
    return array_factor - np.max(array_factor)

def calculate_er(position, antenna, parameters):
    # TODO - hot code
    e_r = np.zeros(parameters.samples, dtype=np.csingle)
    # offset because of calculation funnybusiness... should refactor function
    phi = parameters.phi + np.pi / 2
    phases = np.angle(position)
    weights = np.abs(position)
    for i in range(parameters.samples):
        angle = phi[i]
        theta = antenna.wavenumber * np.cos(angle) * antenna.positions + phases
        e_t = np.sum(weights * np.exp(1j * theta))
        e_r[i] = e_t / antenna.num_elements
    return e_r


def fitness(position, antenna, parameters):
    array_factor = calculate_array_factor(position, antenna, parameters)
    beamwidth = parameters.beamwidth_samples
    peaks, properties = find_peaks(array_factor, height=-50, distance=5)
    score = 0
    for target in parameters.targets:
        beam_range = np.arange(target - beamwidth, target + beamwidth)
        score += np.average(array_factor[beam_range])
    return score


def display(position, antenna, parameters, persist=False):
    array_factor = calculate_array_factor(position, antenna, parameters)
    plt.clf()
    plt.plot(parameters.phi, array_factor)

    targets_markers = (2 * np.pi * parameters.targets / parameters.samples) - np.pi
    for target in targets_markers:
        plt.axvspan(
            target - parameters.beamwidth,
            target + parameters.beamwidth,
            color="green",
            alpha=0.5,
        )

    peaks, _ = find_peaks(array_factor, height=-50, distance=5)
    peak_angles = (2 * np.pi * peaks / parameters.samples) - np.pi
    plt.plot(peak_angles, array_factor[peaks], "X", color="orange")

    # print(f"sidelobe: {sidelobe_level}")
    # plt.axhline(sidelobe_level, color="orange", alpha=0.5)
    plt.xlim(-np.pi / 2, np.pi / 2)
    plt.ylim((-40, 0))
    plt.xlabel("Beam angle [rad]")
    plt.ylabel("Power [dB]")
    if persist:
        plt.show()
    else:
        plt.pause(0.05)


def get_best_position_in_range(population, index, neighbourhood_size):
    best_score = float("-inf")
    best_position = population[index].best_position
    neighbourhood = np.mod(
        np.arange(index - neighbourhood_size, index + neighbourhood_size + 1),
        len(population),
    )
    for i in neighbourhood:
        if population[i].best_score > best_score:
            best_score = population[i].best_score
            best_position = population[i].best_position
    return best_position


def step_PSO(population, global_best_position, global_best_score, antenna, parameters):
    for i, particle in enumerate(population):
        best_known_position = get_best_position_in_range(
            population, i, parameters.neighbourhood_size
        )
        particle.best_known_position = best_known_position
        particle.step(antenna, parameters)
        if particle.score > global_best_score:
            global_best_position = particle.position
            global_best_score = particle.score
    return population, global_best_position, global_best_score


def particle_swarm_optimisation(antenna, parameters, logging):
    population = new_population(antenna, parameters)
    # setting these as an arbitrary member of the initialized population
    global_best_position = population[0].best_position
    global_best_score = population[0].best_score
    for step in range(parameters.max_steps):
        population, global_best_position, global_best_score = step_PSO(
            population, global_best_position, global_best_score, antenna, parameters
        )
        if logging.verbose:
            print(f"Step: {step}")
            print(f"Position: {global_best_position}\n Score: {global_best_score}")
        if logging.show_plots:
            display(global_best_position, antenna, parameters)
    return global_best_position


def beamformer(antenna, parameters, logging):
    result = particle_swarm_optimisation(antenna, parameters, logging)
    if logging.show_plots and logging.plots_persist:
        display(result, antenna, parameters, persist=True)
    return result

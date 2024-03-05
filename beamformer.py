import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences


class Particle:
    def __init__(self, antenna, parameters, fitness_function):
        self.velocity = np.random.uniform(size=antenna.num_elements) * np.exp(
            1j * np.random.uniform(size=antenna.num_elements) * 2 * np.pi
        )
        self.position = np.random.uniform(size=antenna.num_elements) * np.exp(
            1j * np.random.uniform(size=antenna.num_elements) * 2 * np.pi
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
        self.fitness_function = fitness_function

    def step(self):
        """Move the particle to the next position, then update it's score based on the provided fitness function"""
        self.update_velocity()
        self.update_position()
        self.update_score()

    def update_score(self):
        """Update the particle's score based on the fitness function provided"""
        self.score = self.fitness_function(self.position)
        if self.score > self.best_score:
            self.best_position = self.position
            self.best_score = self.score
        if self.score > self.best_known_score:
            self.best_known_position = self.position
            self.best_known_score = self.score

    def update_position(self):
        """Set and limit magnitude (weights) of particle to 1 and wrap its angle (phase) [0, 2pi]"""
        position = np.add(self.position, self.velocity)
        weights = np.clip(np.abs(position), 0, 1)
        phases = np.mod(np.angle(position), 2 * np.pi)
        self.position = weights * np.exp(1j * phases)

    def update_velocity(self):
        """Set and limit velocity of particle to max_particle_velocity and wrap direction [0, 2pi]"""
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
        velocity = inertial_component + cognitive_component + social_component
        speed = np.clip(np.abs(velocity), 0, self.max_velocity)
        direction = np.mod(np.angle(velocity), 2 * np.pi)
        self.velocity = speed * np.exp(1j * direction)

    def set_best_known(self, new_best_known):
        # TODO - should this check if the new_best_known is actually better?
        self.best_known_position = new_best_known.position
        self.best_known_score = new_best_known.score


class Population:
    def __init__(self, antenna, parameters):
        self.population = []
        for i in range(parameters.population_size):
            self.population.append(
                Particle(antenna, parameters, lambda p: fitness(p, antenna, parameters))
            )
        self.neighbourhood_size = parameters.neighbourhood_size
        self.global_best_position = self.population[0].position
        self.global_best_score = self.population[0].score

    def step(self):
        for index, particle in enumerate(
            self.population
        ):  # Update best-knowns for each particle
            best_particle_in_neighbourhood = self.best_particle_in_neighbourhood(index)
            particle.set_best_known(self.population[best_particle_in_neighbourhood])
        for index, particle in enumerate(self.population):  # Step each particle
            particle.step()
            if particle.score > self.global_best_score:
                self.global_best_position = particle.position
                self.global_best_score = particle.score

    def best_particle_in_neighbourhood(self, index):
        neighbourhood = np.mod(
            np.arange(index - self.neighbourhood_size, index + self.neighbourhood_size + 1),
            len(self.population),
        )
        best_score = float("-inf")
        best_particle_index = index
        for i in neighbourhood:
            if self.population[i].best_score > best_score:
                best_particle_index = i
        return best_particle_index


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


def particle_swarm_optimisation(antenna, parameters, logging):
    population = Population(antenna, parameters)
    for step_counter in range(parameters.max_steps):
        population.step()
        if logging.verbose:
            print(f"Step: {step_counter}")
            print(f"Position: {population.global_best_position}\n Score: {population.global_best_score}")
        if logging.show_plots:
            display(population.global_best_position, antenna, parameters)
    return population.global_best_position


def beamformer(antenna, parameters, logging):
    result = particle_swarm_optimisation(antenna, parameters, logging)
    if logging.show_plots and logging.plots_persist:
        display(result, antenna, parameters, persist=True)
    return result

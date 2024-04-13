import numpy as np
from dataclasses import dataclass
from sklearn.cluster import KMeans


class Particle:
    def __init__(self, antenna, parameters, uniform=False):
        if uniform:
            self.velocity = np.zeros(antenna.num_elements) * np.exp(
                1j * np.zeros(antenna.num_elements) * 2 * np.pi
            )
            self.position = np.ones(antenna.num_elements, dtype=complex)
        else:
            self.velocity = np.random.uniform(size=antenna.num_elements) * np.exp(
                1j * np.random.uniform(size=antenna.num_elements) * 2 * np.pi
            )
            self.position = np.random.uniform(size=antenna.num_elements) * np.exp(
                1j * np.random.uniform(size=antenna.num_elements) * 2 * np.pi
            )
        self.best_position = self.position
        self.best_known_position = self.position
        self.clusters = parameters.num_tiles
        self.tiled_position = (
            self.position
        )  # set by generate_tiling() on the first step.
        self.tile_labels = np.zeros(self.clusters, dtype=int)
        self.tile_values = np.zeros(self.clusters, dtype=complex)
        self.score = antenna.fitness(self.tiled_position, parameters)
        self.best_score = self.score
        self.best_known_score = self.score
        self.dimensions = antenna.num_elements
        self.max_velocity = parameters.max_particle_velocity
        self.inertia_weight = parameters.intertia_weight
        self.cognitive_coeff = parameters.cognitive_coeff
        self.social_coeff = parameters.social_coeff

    def generate_tiling(self):
        """Uses k means clustering to update tiled_position, based on phase and amplitude"""
        separated_position = np.zeros((self.dimensions, 2))
        separated_position[:, 0] = np.abs(self.position)
        separated_position[:, 1] = np.angle(self.position)
        kmeans = KMeans(n_clusters=self.clusters).fit(separated_position)
        self.tile_labels = kmeans.labels_
        centres = kmeans.cluster_centers_
        self.tile_values = centres[:, 0] * np.exp(1j * centres[:, 1])
        for index, label in enumerate(self.tile_labels):
            self.tiled_position[index] = self.tile_values[label]

    def step(self, fitness_function):
        """Move the particle to the next position, then update it's score based on the provided fitness function"""
        self.update_velocity()
        self.update_position()
        self.generate_tiling()
        self.update_score(fitness_function)

    def update_score(self, fitness_function):
        """Update the particle's score with the provided fitness function
        Called by step()"""
        self.score = fitness_function(self.tiled_position)
        if self.score > self.best_score:
            self.best_position = self.tiled_position
            self.best_score = self.score
        if self.score > self.best_known_score:
            self.best_known_position = self.tiled_position
            self.best_known_score = self.score

    def update_position(self):
        """Set and limit magnitude (weights) of particle to 1 and wrap its angle (phase) [0, 2pi]
        Called by step()"""
        position = np.add(self.position, self.velocity)
        weights = np.clip(np.abs(position), 0, 1)
        phases = np.mod(np.angle(position), 2 * np.pi)
        self.position = weights * np.exp(1j * phases)

    def random_complex(self, size):
        """Generates a random complex number, sampled from a zero-centred disk"""
        return np.sqrt(np.random.uniform(0, 1, size)) * np.exp(
            1.0j * np.random.uniform(0, 2 * np.pi, size)
        )

    def update_velocity(self):
        """Set and limit velocity of particle to max_particle_velocity and wrap direction [0, 2pi]
        Called by step()"""
        # TODO should the random numbers here be complex as well? or just reals?
        inertial_component = self.inertia_weight * self.velocity
        cognitive_component = (
            self.cognitive_coeff
            * self.random_complex(self.dimensions)
            * (np.subtract(self.best_position, self.position))
        )
        social_component = (
            self.social_coeff
            * self.random_complex(self.dimensions)
            * (np.subtract(self.best_known_position, self.position))
        )
        # TODO add component that influences particles toward their tiled position
        velocity = inertial_component + cognitive_component + social_component
        speed = np.clip(np.abs(velocity), 0, self.max_velocity)
        direction = np.mod(np.angle(velocity), 2 * np.pi)
        self.velocity += speed * np.exp(1j * direction)

    def set_best_known(self, new_best_known):
        """Update the particle with a best_known position, distinct from the particle's own best position"""
        # TODO - should this check if the new_best_known is actually better?
        self.best_known_position = new_best_known.position
        self.best_known_score = new_best_known.score


class Population:
    def __init__(self, antenna, parameters):
        self.population = []
        for i in range(parameters.population_size):
            self.population.append(Particle(antenna, parameters))
        self.neighbourhood_size = parameters.neighbourhood_size
        self.global_best_position = self.population[0].position
        self.global_best_score = self.population[0].score
        self.global_best_tiled_position = self.population[0].tiled_position
        self.fitness_function = lambda p: antenna.fitness(p, parameters)

    def step(self):
        """Take a single step in the simulation, update all particles once and recalculate best position"""
        # First update each particle's best-known position
        for index, particle in enumerate(self.population):
            best_particle_in_neighbourhood = self.best_particle_in_neighbourhood(index)
            particle.set_best_known(self.population[best_particle_in_neighbourhood])
        # Then step() each particle, updating their scores
        for index, particle in enumerate(self.population):
            particle.step(self.fitness_function)
            if particle.score > self.global_best_score:
                self.global_best_position = particle.position
                self.global_best_score = particle.score

    def best_particle_in_neighbourhood(self, index):
        """Find the best scoring particle in the neighbourhood (by particle index) of a given particle"""
        neighbourhood = np.mod(
            np.arange(
                index - self.neighbourhood_size, index + self.neighbourhood_size + 1
            ),
            len(self.population),
        )
        best_score = float("-inf")
        best_particle_index = index
        for i in neighbourhood:
            if self.population[i].best_score > best_score:
                best_particle_index = i
        return best_particle_index


def particle_swarm_optimisation(antenna, parameters, logging):
    population = Population(antenna, parameters)
    result = []
    for step_counter in range(parameters.max_steps):
        population.step()
        result.append(
            {
                "best_position_history": population.global_best_position,
                "best_tiled_position_history": population.global_best_tiled_position,
                "best_score_history": population.global_best_score,
            }
        )
        if logging.verbose:
            print(f"Step: {step_counter}/{parameters.max_steps-1}")
            print(
                f"Position: {population.global_best_position}\n Score: {population.global_best_score}"
            )
        if logging.show_plots:
            antenna.display(
                population.global_best_position, population.global_best_tiled_position
            )
    return result


def beamformer(antenna, parameters, logging, config_name):
    if logging.debug:
        debug_particle = Particle(antenna, parameters, uniform=True)
        antenna.display(debug_particle.position, debug_particle.position, persist=True)
        return None
    else:
        if logging.verbose:
            print(f"Starting simulation: {config_name}")
        result = particle_swarm_optimisation(antenna, parameters, logging)
        if logging.verbose:
            print("Simulation stopped")
        if logging.show_plots:
            antenna.display(
                result[-1]["best_position_history"],
                result[-1]["best_tiled_position_history"],
                persist=logging.plots_persist,
            )
        return result

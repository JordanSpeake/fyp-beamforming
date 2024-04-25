import numpy as np
from sklearn.cluster import KMeans
import bf_utils


class Particle:
    def __init__(self, antenna, parameters, uniform=False):
        if uniform:
            self.velocity = np.zeros(antenna.num_elements, dtype=complex)
            self.position = np.ones(antenna.num_elements, dtype=complex)
        else:
            self.velocity = bf_utils.random_complex(antenna.num_elements)
            self.position = bf_utils.random_complex(antenna.num_elements)
        self.best_position = self.position
        self.best_neighbour = None
        self.tile_labels = np.zeros(antenna.num_elements, dtype=int)
        self.tile_values = np.zeros(parameters.num_tiles, dtype=complex)
        self.tiled_position = np.zeros(antenna.num_elements, dtype=complex)
        self.score = float("-inf")
        self.best_score = self.score


class Population:
    def __init__(self, antenna, parameters, uniform=False):
        self.population = []
        for i in range(parameters.population_size):
            self.population.append(Particle(antenna, parameters, uniform=uniform))
        self.neighbourhood_size = parameters.neighbourhood_size

        self.fitness_function = lambda p: antenna.fitness(p)
        self.inertia_weight = parameters.intertia_weight
        self.cognitive_coeff = parameters.cognitive_coeff
        self.social_coeff = parameters.social_coeff
        self.max_velocity = parameters.max_particle_velocity
        self.phase_bit_depth = parameters.phase_bit_depth

        self.num_elements = antenna.num_elements
        self.clusters = parameters.num_tiles

        self.elitism_count = parameters.elitism_count
        self.elitism_replacement_chance = parameters.elitism_replacement_chance
        self.best_particles = self.population[0 : self.elitism_count]

    def quantize_phase(self, phase):
        bits = np.power(2, self.phase_bit_depth)
        quantisation_step = int((phase / 2 * np.pi) * bits)
        phase = quantisation_step * 2 * np.pi / bits
        return phase

    def update_velocity(self, particle):
        """Set and limit velocity of particle to max_particle_velocity and wrap direction [0, 2pi]"""
        inertial_component = self.inertia_weight * particle.velocity
        cognitive_component = (
            self.cognitive_coeff
            * bf_utils.random_complex(self.num_elements)
            * (np.subtract(particle.best_position, particle.position))
        )
        social_component = (
            self.social_coeff
            * bf_utils.random_complex(self.num_elements)
            * (np.subtract(particle.best_neighbour.position, particle.position))
        )
        velocity = inertial_component + cognitive_component + social_component
        speed = np.clip(np.abs(velocity), 0, self.max_velocity)
        direction = np.mod(np.angle(velocity), 2 * np.pi)
        particle.velocity = speed * np.exp(1j * direction)

    def update_position(self, particle):
        """Set and limit magnitude (weights) of particle to 1 and wrap its angle (phase) [0, 2pi]"""
        position = np.add(particle.position, particle.velocity)
        weights = np.clip(np.abs(position), 0, 1)
        phases = np.mod(np.angle(position), 2 * np.pi)
        if self.phase_bit_depth > 0:
            for index, phase in enumerate(phases):
                phases[index] = self.quantize_phase(phase)
        particle.position = weights * np.exp(1j * phases)

    def generate_tiling(self, particle):
        """Uses k means clustering to update tiled_position, based on phase and amplitude"""
        separated_position = np.zeros((self.num_elements, 2))
        separated_position[:, 0] = np.abs(particle.position)
        separated_position[:, 1] = np.angle(particle.position)
        kmeans = KMeans(n_clusters=self.clusters).fit(separated_position)
        particle.tile_labels = kmeans.labels_
        centres = kmeans.cluster_centers_
        particle.tile_values = centres[:, 0] * np.exp(1j * centres[:, 1])
        for index, label in enumerate(particle.tile_labels):
            particle.tiled_position[index] = particle.tile_values[label]

    def generate_best_particle_list(self):
        best_particles = []
        for particle in self.population:
            if len(best_particles) == 0:
                best_particles.append(particle)
            elif len(best_particles) < self.elitism_count:
                for index, particle_in_list in enumerate(best_particles):
                    if particle.score > particle_in_list.score:
                        best_particles.insert(index, particle)
                        break
                best_particles.append(particle)
            else:
                for index, particle_in_list in enumerate(self.best_particles):
                    if particle.score > particle_in_list.score:
                        self.best_particles.pop(-1)
                        self.best_particles.insert(index, particle)
                        break

    def update_particle(self, particle, fitness_function):
        """Move the particle to the next position, then update it's score based on the provided fitness function"""
        self.update_velocity(particle)
        self.update_position(particle)
        self.generate_tiling(particle)
        particle.score = fitness_function(particle.tiled_position)

    def best_neighbour_index(self, index):
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
            if self.population[i].score > best_score:
                best_particle_index = i
        return best_particle_index

    def update_best_neighbours(self):
        for index, particle in enumerate(self.population):
            particle.best_neighbour = self.population[self.best_neighbour_index(index)]

    def repopulate_with_elitism(self):
        """Replace a random selection of low-scoring particles with high scoring particles"""
        for particle in self.population:
            if particle not in self.best_particles:
                if np.random.rand() < self.elitism_replacement_chance:
                    new_particle_index = np.random.randint(0, self.elitism_count)
                    particle.position = self.best_particles[new_particle_index].position
                    particle.velocity = self.best_particles[
                        new_particle_index
                    ].velocity + (
                        self.inertia_weight * bf_utils.random_complex(self.num_elements)
                    )

    def step(self):
        """Take a single step in the simulation, update all particles once."""
        self.update_best_neighbours()
        for particle in self.population:
            self.update_particle(particle, self.fitness_function)
        self.generate_best_particle_list()
        self.repopulate_with_elitism()


class PSO:
    def __init__(self, antenna, parameters, logging):
        self.antenna = antenna
        self.parameters = parameters
        self.logging = logging
        self.population = Population(
            self.antenna, self.parameters, uniform=self.logging.use_uniform_particle
        )
        self.result = []

    def update_results(self):
        self.result.append(
            {
                "Position": self.population.best_particles[0].position,
                "Tiled Position": self.population.best_particles[0].tiled_position,
                "Score": self.population.best_particles[0].score,
            }
        )

    def run(self):
        for step_counter in range(self.parameters.max_steps):
            self.population.step()
            self.update_results()
            if self.logging.verbose:
                print(f"Step: {step_counter}/{self.parameters.max_steps-1}")
                print(
                    f"Position: {self.population.best_particles[0].position}\n Score: {self.population.best_particles[0].score}"
                )
            if self.logging.show_plots:
                self.antenna.display(
                    self.population.best_particles[0].position,
                    self.population.best_particles[0].tiled_position,
                    self.population.best_particles[0].tile_labels,
                )
        return self.result


def beamformer(antenna, parameters, logging, config_name):
    particle_swamp_optimiser = PSO(antenna, parameters, logging)
    if logging.verbose:
        print(f"Starting simulation: {config_name}")
    result = particle_swamp_optimiser.run()
    if logging.verbose:
        print("Done.")
    if logging.show_plots:
        antenna.display(
            result[-1]["Position"],
            result[-1]["Tiled Position"],
            persist=logging.plots_persist,
        )
    return result

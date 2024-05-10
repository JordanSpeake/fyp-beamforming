import bf_utils
from sklearn.cluster import KMeans
import numpy as np


class Particle:
    def __init__(self, params, antenna, subswarm):
        self.subswarm = subswarm
        self.num_elements = antenna.num_elements
        self.num_clusters = params.num_clusters
        self.velocity = (
            bf_utils.random_complex(antenna.num_elements) * params.max_particle_velocity
        )
        self.position = subswarm.centroid + (
            bf_utils.random_complex(antenna.num_elements) * params.subswarm_init_radius
        )
        self.tile_labels = np.zeros(antenna.num_elements, dtype=int)
        self.tile_values = np.zeros(params.num_clusters, dtype=complex)
        self.tiled_position = np.zeros(antenna.num_elements, dtype=complex)
        self.objective_function = lambda position: antenna.fitness(position)
        self.inertia_weight = params.particle_inertia_weight
        self.phase_bit_depth = params.phase_bit_depth
        self.max_particle_velocity = params.max_particle_velocity
        self.centroid_velocity_coeff = params.centroid_velocity_coeff
        self.social_coeff = params.social_coeff
        self.score = None
        self.islr = None
        self.mle = None
        self.mle_sum = None
        self.sle = None
        self.mse = None
        self.update_score()

    def update_position(self):
        """Update the particle's position by addition of it's velocity"""
        position = np.add(self.position, self.velocity)
        weights = np.clip(np.abs(position), 0, 1)
        phases = np.mod(np.angle(position), 2 * np.pi)
        if self.phase_bit_depth > 0:
            for index, phase in enumerate(phases):
                phases[index] = bf_utils.quantize(phase, self.phase_bit_depth)
        self.position = weights * np.exp(1j * phases)

    def update_velocity(self):
        """Update the particle's velocity, and limit it to self.max_particle_velocity"""
        inertial_component = self.velocity * self.inertia_weight
        toward_centroid_component = (
            self.subswarm.score_weighted_centroid - self.position
        ) * self.centroid_velocity_coeff
        social_component = (
            self.subswarm.best_particle.position - self.position
        ) * self.social_coeff
        subswarm_velocity_component = self.subswarm.velocity
        random_component = (
            0.001 * self.islr * bf_utils.random_complex(self.num_elements)
        )
        self.velocity = (
            toward_centroid_component + subswarm_velocity_component + inertial_component
        )
        velocity_magnitude = np.sqrt(np.dot(self.velocity, self.velocity))
        if velocity_magnitude > self.max_particle_velocity:
            self.velocity = np.multiply(
                np.divide(self.velocity, velocity_magnitude), self.max_particle_velocity
            )

    def generate_tiling(self):
        """Uses k means clustering to update tiled_position, based on real and imag values"""
        separated_position = np.zeros((self.num_elements, 2))
        separated_position[:, 0] = np.real(self.position)
        separated_position[:, 1] = np.imag(self.position)
        kmeans = KMeans(n_clusters=self.num_clusters).fit(separated_position)
        self.tile_labels = kmeans.labels_
        centres = kmeans.cluster_centers_
        self.tile_values = centres[:, 0] * np.exp(1j * centres[:, 1])
        for index, label in enumerate(self.tile_labels):
            self.tiled_position[index] = self.tile_values[label]

    def update_score(self):
        if self.num_clusters <= 0:
            mse, islr, doi_mle, dnoi_mle, sle, af_difference = self.objective_function(
                self.position
            )
        else:
            self.generate_tiling()
            mse, islr, doi_mle, dnoi_mle, sle, af_difference = self.objective_function(
                self.tiled_position
            )
        self.islr = islr
        self.doi_mle = doi_mle
        self.dnoi_mle = dnoi_mle
        self.mle_sum = np.sum(doi_mle)
        self.sle = sle
        self.mse = mse
        # Use self.score to set how the score of a particle is evaluated.
        # This should ideally be set using the config file, but that wasn't implemented.
        self.score = 1 / mse

    def step(self):
        self.update_velocity()
        self.update_position()
        self.update_score()


class SubSwarm:
    def __init__(self, params, antenna, swarm):
        self.centroid = bf_utils.random_complex(antenna.num_elements)
        self.velocity = np.zeros(antenna.num_elements, dtype=complex)
        self.particles = [
            Particle(params, antenna, self) for _ in range(params.subswarm_size)
        ]
        self.score_weighted_centroid = self.calculate_score_weighted_centroid()
        self.subswarm_charge_coeff = params.subswarm_charge
        self.swarm = swarm
        self.best_particle = self.particles[0]
        self.objective_function = lambda position: antenna.fitness(position)
        self.rerandomisation_proximity = params.rerandomisation_proximity

    def calculate_centroid(self):
        positions = [particle.position for particle in self.particles]
        return np.mean(positions, axis=0)

    def calculate_score_weighted_centroid(self):
        positions = [particle.position for particle in self.particles]
        scores = [particle.score for particle in self.particles]
        return np.average(positions, weights=scores, axis=0)

    def calculate_coulomb_force(self):
        resultant_force = 0
        for subswarm in self.swarm.subswarms:
            if subswarm != self:
                distance = np.linalg.norm(self.centroid - subswarm.centroid)
                direction = np.divide(self.centroid - subswarm.centroid, distance)
                resultant_force += direction * np.divide(
                    distance, np.power(distance, 3)
                )
        return resultant_force

    def update_velocity(self):
        self.centroid = self.calculate_centroid()
        self.score_weighted_centroid = self.calculate_score_weighted_centroid()
        coulomb_velocity = self.calculate_coulomb_force() * self.subswarm_charge_coeff
        self.velocity = coulomb_velocity

    def check_for_rerandomisation(self):
        """If a subswarm is too close to another, re-randomise the worse performing subswarm"""
        for subswarm in self.swarm.subswarms:
            if subswarm != self:
                distance = np.linalg.norm(self.centroid - subswarm.centroid)
                if distance <= self.rerandomisation_proximity:
                    own_score, _, _, _ = self.objective_function(
                        self.score_weighted_centroid
                    )
                    other_score, _, _, _ = self.objective_function(
                        subswarm.score_weighted_centroid
                    )
                    score_difference = (1 / own_score) - (1 / other_score)
                    if score_difference < 0:
                        self.swarm.rerandomise_subswarm(self)

    def step(self):
        self.check_for_rerandomisation()
        self.update_velocity()
        for particle in self.particles:
            particle.step()
            if particle.score > self.best_particle.score:
                self.best_particle = particle


class Swarm:
    def __init__(self, params, antenna):
        self.subswarms = [
            SubSwarm(params, antenna, self) for _ in range(params.num_subswarms)
        ]  # a list of the subswarms within the swarm, maintained as a sorted list by the centroid's score
        self.best_particle = self.subswarms[0].particles[0]
        self.params = params
        self.antenna = antenna

    def step(self):
        for subswarm in self.subswarms:
            subswarm.step()
            if subswarm.best_particle.score > self.best_particle.score:
                self.best_particle = subswarm.best_particle

    def rerandomise_subswarm(self, subswarm):
        subswarm.centroid = bf_utils.random_complex(self.antenna.num_elements) * (
            1 - self.params.max_particle_velocity
        )
        subswarm.velocity = np.zeros(self.antenna.num_elements, dtype=complex)
        subswarm.particles = [
            Particle(self.params, self.antenna, subswarm)
            for _ in range(self.params.subswarm_size)
        ]


def print_particle_stats(particle):
    # print(f"    Best Position: {swarm.best_particle.position}")
    print(f"    Score: {particle.score}")
    print(f"    ISLR: {particle.islr}, {bf_utils.to_dB(particle.islr)} dB")
    print(f"    SLE: {particle.sle}, {bf_utils.to_dB(particle.sle)} dB")
    print(f"    MLEs: {particle.mle}, {bf_utils.to_dB(particle.mle)} dB")


def plot_particle_data(antenna, particle):
    antenna.display(
        particle.position,
        particle.tiled_position,
        particle.tile_labels,
    )


def get_results_from_particle(particle):
    return {
        "position": particle.position,
        "tiled_position": particle.tiled_position,
        "score": particle.score,
        "islr": particle.islr,
        "mle_sum": particle.mle_sum,
        "mle": particle.mle,
        "sle": particle.sle,
    }


def print_swarm_stats(swarm):
    print("Subswarms:")
    for index, subswarm in enumerate(swarm.subswarms):
        _, islr, _, _, _ = subswarm.objective_function(subswarm.score_weighted_centroid)
        print(f"    SS{index} ISLR: {islr}, {bf_utils.to_dB(islr)}")


def beamformer(antenna, params, logging, config_name):
    if logging.verbose:
        print("Setting up... ", end=" ", flush=True)
    swarm = Swarm(params, antenna)
    if logging.verbose:
        print("Done.")
    results = []
    for step in range(params.max_steps):
        if logging.verbose:
            print(f"Step {step}/{params.max_steps-1}")
        swarm.step()
        if logging.verbose:
            print_particle_stats(swarm.best_particle)
            print_swarm_stats(swarm)
        if logging.show_plots:
            plot_particle_data(antenna, swarm.best_particle)
        results.append(get_results_from_particle(swarm.best_particle))
    if logging.output_final_values:
        print(
            f"DOI MLEs: {swarm.best_particle.doi_mle}\n DNOI MLEs: {swarm.best_particle.dnoi_mle}"
        )
    return results

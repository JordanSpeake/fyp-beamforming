import beamformer as bf
import numpy as np
import setup_utils


def main():
    antenna = setup_utils.define_ULA(2e9, 0.5, 15)
    parameters = setup_utils.define_parameters(
        population_size=100,
        angular_samples=360 * 4,
        cognitive_coeff=0.1,
        social_coeff=0.1,
        intertia_weight=0.1,
        max_steps=500,
        static_targets=[1],
        beamwidth=0.05, # good range 0.01 to 0.1
        sidelobe_suppression=10,  # good range 10-100
        max_particle_velocity=1,  # Per dimension
        neighbourhood_size=10, # for local PSO topology, how many neighbours in each direction a particle knows about
    )
    logging = setup_utils.define_logging(show_plots=True, plots_persist=True, verbose=True)
    result = bf.beamformer(antenna, parameters, logging)


main()

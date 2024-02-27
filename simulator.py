import beamformer as bf
import numpy as np


def main():
    antenna = bf.define_ULA(2e9, 0.5, 20)
    parameters = bf.define_parameters(
        population_size=100,
        angular_samples=360 * 4,
        cognitive_coeff=0.35,
        social_coeff=0.35,
        intertia_weight=0.8,
        max_steps=50,
        static_targets=np.linspace(-np.pi / 2 + 0.5, np.pi / 2 - 0.5, 0),
        beamwidth=0.02,
        sidelobe_suppression=50,  # good range 30-100
        max_particle_velocity=5,  # Per dimension
    )
    logging = bf.define_logging(show_plots=True, plots_persist=True, verbose=True)
    result = bf.beamformer(antenna, parameters, logging)


main()

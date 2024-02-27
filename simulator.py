import beamformer as bf

def main():
    antenna = bf.define_ULA(2e9, 0.5, 15)
    static_targets = [1, -1] # in radians
    population_size = 100
    angular_samples = 1000
    intertia_weight = 0.9
    cognitive_coeff = 0.5
    social_coeff = 0.5
    max_steps = 50
    parameters = bf.define_parameters(population_size, angular_samples, cognitive_coeff, social_coeff, intertia_weight, max_steps, static_targets)
    logging = bf.define_logging(show_plots=True, plots_persist=True)
    result = bf.beamformer(antenna, parameters, logging)

main()

import beamformer as bf

def main():
    antenna = bf.define_ULA(2e9, 0.5, 10)
    static_targets = [1, -1] # in radians
    population_size = 100
    angular_samples = 1500
    intertia_weight = 0.5
    cognitive_coeff = 0.8
    social_coeff = 0.8
    parameters = bf.define_parameters(population_size, angular_samples, cognitive_coeff, social_coeff, intertia_weight, 25, static_targets)
    result = bf.beamformer(antenna, parameters)

main()

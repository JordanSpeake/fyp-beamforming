import beamformer as bf

def main():
    antenna = bf.define_ULA(2e9, 0.5, 10)
    parameters = bf.define_parameters(1800, 25)
    result = bf.beamformer(antenna, parameters)
    print(result)

main()

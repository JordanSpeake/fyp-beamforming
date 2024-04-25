"""The entry-point for the beamforming simulator
usage: python simulator.py --config path/to/config.toml"""

import argparse
import csv
import datetime
import os.path
import cProfile

try:
    import tomlib
except ImportError:
    import tomli as tomlib
import beamformer as bf
import bf_utils
import antennas


def read_config(path_to_config):
    """Parses the provided TOML file, generating classes required for the simulation"""
    with open(path_to_config, "rb") as file:
        data = tomlib.load(file)
        parameters = parse_parameters_config(data["parameters"])
        if parameters is None:
            return None
        antenna = parse_antenna_config(data["antenna"], parameters)
        if antenna is None:
            return None
        logging = parse_logging_config(data["logging"])
        if logging is None:
            return None
        return {
            "antenna": antenna,
            "parameters": parameters,
            "logging": logging,
            "config_name" : os.path.basename(path_to_config)
        }


def parse_antenna_config(data, parameters_config):
    """Constructs an instants of an Antenna child class defined in [antenna] in the config file"""
    try:
        if data["type"] == "UniformLinear":
            return antennas.UniformLinear(
                data["frequency"],
                data["spacing"],
                data["num_elements"],
                parameters_config,
            )
        if data["type"] == "Circular":
            return antennas.Circular(
                data["frequency"],
                data["radius"],
                data["num_elements"],
                parameters_config,
            )
        if data["type"] == "RectangularPlanar":
            return antennas.RectangularPlanar(
                data["frequency"],
                data["spacing"],
                data["num_elements"],
                parameters_config,
            )
    except KeyError as e:
        print(f"Failed to parse antenna config: {e}")
        return None
    print("No/invalid specified antenna")
    return None


def parse_parameters_config(data):
    """Constructs an instants of the parameters calss defined in [parameters] in the config file"""
    try:
        parameters = bf_utils.Parameters(
            population_size=data["population_size"],
            samples=data["samples"],
            cognitive_coeff=data["cognitive_coeff"],
            social_coeff=data["social_coeff"],
            intertia_weight=data["inertia_weight"],
            max_steps=data["max_steps"],
            static_targets=data["static_targets"],
            max_particle_velocity=data["max_particle_velocity"],
            neighbourhood_size=data["neighbourhood_size"],
            num_tiles=data["num_tiles"],
            phase_bit_depth=data["phase_bit_depth"],
            elitism_count=data["elitism_count"],
            elitism_replacement_chance=data["elitism_replacement_chance"],
        )
    except KeyError as e:
        print(f"Failed to parse parameters config: {e}")
        return None
    return parameters


def parse_logging_config(data):
    """Constructs an instants of the Logging class defined in [logging] in the config file"""
    try:
        logging = bf_utils.Logging(
            show_plots=data["show_plots"],
            plots_persist=data["plots_persist"],
            verbose=data["verbose"],
            write_results=data["write_results"],
            use_uniform_particle=data["use_uniform_particle"],
        )
    except KeyError as e:
        print(f"Failed to parse logging config: {e}")
        return None
    return logging


def parse_config(args):
    """Returns a parsed config file from the provided arguments"""
    try:
        path_to_config = args.config[0]
    except SystemExit:
        print("Invalid path to config file")
        return None
    config = read_config(path_to_config)
    if config is None:
        print(f"Error in config file: {path_to_config}")
        return None
    return config


def get_output_path(config_name):
    """Constructs the path and name for the output .csv file"""
    datetime_marker = datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S")
    output_path = f"./{config_name}_{datetime_marker}.csv"
    return output_path



def generate_simulator_setup():
    """Generate the required classes for running the simulation"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        nargs=1,
        required=True,
        help="--config ./path/to/config.toml",
    )
    arguments = parser.parse_args()
    config = parse_config(arguments)
    return (
        config["antenna"],
        config["parameters"],
        config["logging"],
        config["config_name"],
    )


def write_results(bf_result, file):
    """Write the results from beamformer.py to the specified file"""
    writer = csv.DictWriter(file, fieldnames=bf_result[0].keys(), dialect="excel")
    writer.writeheader()
    writer.writerows(bf_result)


def main():
    antenna, parameters, logging, config_name = generate_simulator_setup()
    if logging.write_results:
        output_path = get_output_path(config_name)
        with open(output_path, "w", newline="", encoding="utf-8") as file:
            try:
                result = bf.beamformer(antenna, parameters, logging, config_name)
                result.append()
            except Exception as e:
                print(f"Simulation cancelled, error in beamformer.py: {e}")
            else:
                write_results(result, file)
            print("Simulation results written successfully")
    else:
        with cProfile.Profile() as pr:
            _ = bf.beamformer(antenna, parameters, logging, config_name)
            pr.dump_stats('import_debug_profile_dump')
    print("Done.")


main()

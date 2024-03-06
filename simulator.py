import beamformer as bf
import numpy as np
import argparse, setup_utils, antennas

try:
    import tomlib
except ImportError:
    import tomli as tomlib


def parse_config(path_to_config):
    with open(path_to_config, "rb") as file:
        data = tomlib.load(file)
        antenna = parse_antenna_config(data["antenna"])
        if antenna is None:
            return None
        parameters = parse_parameters_config(data["parameters"])
        if parameters is None:
            return None
        logging = parse_logging_config(data["logging"])
        if logging is None:
            return None
        return {
            "antenna": antenna,
            "parameters": parameters,
            "logging": logging,
        }


def parse_antenna_config(data):
    if data["type"] == "ULA":
        try:
            antenna = antennas.ULA(
                data["frequency"], data["spacing_coeff"], data["num_elements"]
            )
        except KeyError:
            print("Failed to parse antenna config")
            return None
    return antenna


def parse_parameters_config(data):
    try:
        parameters = setup_utils.Parameters(
            population_size=data["population_size"],
            samples=data["samples"],
            cognitive_coeff=data["cognitive_coeff"],
            social_coeff=data["social_coeff"],
            intertia_weight=data["inertia_weight"],
            max_steps=data["max_steps"],
            static_targets=data["static_targets"],
            beamwidth=data["beamwidth"],
            max_particle_velocity=data["max_particle_velocity"],
            neighbourhood_size=data["neighbourhood_size"],
        )
    except KeyError:
        print("Failed to parse parameters config")
        return None
    return parameters


def parse_logging_config(data):
    try:
        logging = setup_utils.Logging(
            show_plots=data["show_plots"],
            plots_persist=data["plots_persist"],
            verbose=data["verbose"],
        )
    except KeyError:
        print("Failed to parse logging config")
        return None
    return logging


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        nargs=1,
        required=True,
        help="--config ./path/to/config.toml",
    )
    try:
        path_to_config = parser.parse_args().config[0]
    except SystemExit:
        print("Invalid path to config file")
        return
    config = parse_config(path_to_config)
    if config is None:
        print(f"Error in config file: {path_to_config}")
        return
    result = bf.beamformer(config["antenna"], config["parameters"], config["logging"])


main()

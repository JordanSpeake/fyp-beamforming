import beamformer as bf
import numpy as np
import argparse, setup_utils, antennas, csv, datetime

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
        try:
            config_name = data["config_name"]
        except KeyError as e:
            config_name = "untitled_config"
        return {
            "config_name": config_name,
            "antenna": antenna,
            "parameters": parameters,
            "logging": logging,
        }


def parse_antenna_config(data):
    try:
        if data["type"] == "UniformLinear":
            return antennas.UniformLinear(
                data["frequency"], data["spacing"], data["num_elements"]
            )
        elif data["type"] == "Circular":
            return antennas.Circular(
                data["frequency"], data["radius"], data["num_elements"]
            )
    except KeyError as e:
        print(f"Failed to parse antenna config: {e}")
        return None
    print(f"No/invalid specified antenna")
    return None


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
    except KeyError as e:
        print(f"Failed to parse parameters config: {e}")
        return None
    return parameters


def parse_logging_config(data):
    try:
        logging = setup_utils.Logging(
            show_plots=data["show_plots"],
            plots_persist=data["plots_persist"],
            verbose=data["verbose"],
            debug=data["debug"],
        )
    except KeyError as e:
        print("Failed to parse logging config: {e}")
        return None
    return logging


def get_config(args):
    try:
        path_to_config = args.config[0]
    except SystemExit:
        print("Invalid path to config file")
        return None
    config = parse_config(path_to_config)
    if config is None:
        print(f"Error in config file: {path_to_config}")
        return None
    return config


def get_output_path(config_name):
    datetime_marker = datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S")
    output_path = f"./{config_name}_{datetime_marker}.csv"
    return output_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        nargs=1,
        required=True,
        help="--config ./path/to/config.toml",
    )
    arguments = parser.parse_args()
    config = get_config(arguments)
    if config is None:
        return
    antenna = config["antenna"]
    parameters = config["parameters"]
    logging = config["logging"]
    config_name = config["config_name"]
    if not logging.debug:
        output_path = get_output_path(config["config_name"])
        with open(output_path, "w", newline="", encoding="utf-8") as file:
            try:
                result = bf.beamformer(antenna, parameters, logging, config_name)
            except Exception as e:
                print(f"Simulation cancelled, error in beamformer.py: {e}")
            else:
                writer = csv.DictWriter(
                    file, fieldnames=result[0].keys(), dialect="excel"
                )
                writer.writeheader()
                writer.writerows(result)
            print(f"Simulation results written successfully")
    else:
        print(f"DEBUG: {config_name}")
        _ = bf.beamformer(antenna, parameters, logging, config_name)
    print("Done.")


main()

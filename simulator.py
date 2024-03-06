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
            "config_name" : config_name,
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
        except KeyError as e:
            print(f"Failed to parse antenna config: {e}")
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

def touch_output_file(config_name):
    datetime_marker = datetime.datetime.now().strftime("%Y%m%d-%H_%M_%S")
    output_path = f"./{config_name}_{datetime_marker}.csv"
    with open(output_path, 'w') as file:
        writer = csv.writer(file, dialect='excel')
    return output_path

def export_result(result, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=result[0].keys(), dialect="excel")
        writer.writeheader()
        writer.writerows(result)

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
    if config is None: return
    output_path = touch_output_file(config["config_name"])
    result = bf.beamformer(config["antenna"], config["parameters"], config["logging"])
    export_result(result, output_path)

main()

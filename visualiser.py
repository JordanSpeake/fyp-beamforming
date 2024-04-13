import numpy as np
import argparse, csv, antennas


def read_results(args):
    try:
        path_to_result = args.file[0]
    except SystemExit:
        print("Invalid path to results file")
        return None
    return parse_results(path_to_result)


def parse_results(path):
    with open(path) as file:
        reader = csv.DictReader(file, dialect="excel")
        for row in reader:
            print(row["best_score_history"])


def display(results):
    # Create a plot, displaying the tiled and untiled position.,
    tiled_positions = results["best_tiled_position_history"]
    positions = results["best_position_history"]
    scores = results["best_score_history"]

    # Also included should be the tiling pattern itself
    # Also display the score
    # Animate, stepping through each result, holding on the final.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        nargs=1,
        required=True,
        help="--file ./path/to/result.csv",
    )
    arguments = parser.parse_args()
    results = read_results(arguments)
    display(results)


main()

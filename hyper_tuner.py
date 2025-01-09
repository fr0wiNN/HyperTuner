"""
Module Name: hyper_tuner.py
Author: ...
Description:
    This script is an entry point for hyperparameter tuning.
    To produce most efficient training parameters, tool uses Successive Halving method which should drastically reduces time for finding the optimal config.

Usage:
    python hyper_tuner.py <tuner_config.yaml>

Project Dependencies:
    - PyYAML ('pip install pyyaml')
    - numpy

Licence:
    MIT License
"""

import sys
import os
import config_manager as cm
from training_manager import TrainingManager

def main():
    """
    Main function that orchestrates the script execution flow.

    Reads the YAML file passed as a command-line argument, processes its content and starts parameters optimization.
    At the end of the execution, prints out the optimal config.
    """
    if len(sys.argv) != 2:
        print("Usage: python hyper_tuner.py <tuner_config.yaml>")
        sys.exit(1)

    tuner_yaml_file = sys.argv[1]
    tuner_config = cm.load_yaml(tuner_yaml_file)

    # Ensure the experiments directory exists
    experiments_dir = "./experiments"
    os.makedirs(experiments_dir, exist_ok=True)

    # Get the next experiment number by counting existing experiment_x folders
    existing_experiments = [d for d in os.listdir(experiments_dir) if d.startswith("experiment_")]
    experiment_number = len(existing_experiments)

    # Define experiment directory and subdirectories
    experiment_dir = os.path.join(experiments_dir, f"experiment_{experiment_number}")
    population_dir = os.path.join(experiment_dir, "population")
    results_dir = os.path.join(experiment_dir, "results")

    # Create the directories
    os.makedirs(population_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model_trainer = TrainingManager(tuner_config)
    model_trainer.experiment_number = experiment_number

    model_trainer.find_best_config()



if __name__ == "__main__":
    main()
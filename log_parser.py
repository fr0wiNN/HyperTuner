"""
Module Name: hyper_tuner.py
Author: ...
Description:
    This script is an entry point for hyperparameter tuning.
    To produce most efficient training parameters, tool uses Successive Halving method which drastically reduces time for finding the optimal config.

Usage:
    python hyper_tuner.py tuner_config.yaml

Dependencies:
    - PyYAML ('pip install pyyaml')

Licence:
    MIT License
"""

import re

# Patterns of log file structure, had to use GPT :(
log_pattern = re.compile(
    r"Step:\s(?P<step>\d+).*?"
    r"Time Elapsed:\s(?P<time_elapsed>[\d.]+)\s?s.*?"
    r"Mean Reward:\s(?P<mean_reward>[-\d.]+).*?"
    r"Mean Group Reward:\s(?P<mean_group_reward>[-\d.]+).*?"
)

class LogParser:
    def __init__(self):
        print("created LogParser")

    def parse(self, training_log):
        """
        Parses the training log and extracts step-wise data for heuristic calculation.

        :param training_log: Raw log output as a string.
        :return: List of dictionaries containing structured training data.
        """
        parsed_data = []

        for line in training_log.split("\n"):
            match = log_pattern.search(line)
            if match:
                step_data = {
                    "step": int(match.group("step")),
                    "time_elapsed": float(match.group("time_elapsed").strip(".")),
                    "mean_reward": float(match.group("mean_reward").strip(".")),
                    "mean_group_reward": float(match.group("mean_group_reward").strip(".")),
                }
                parsed_data.append(step_data)

        return parsed_data

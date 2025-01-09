"""
Module Name: heuristic_calculator.py
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

import numpy as np

class HeuristicCalculator:
    def __init__(self):
        print("created HeuristicCalculator")

    def calculate(self, parsed_log):
        """
        Computes a heuristic score for a single training configuration based on its parsed log.

        :param parsed_log: List of dictionaries representing log entries for a single configuration.
        :return: Heuristic score (float) for the given configuration.
        """
        if not parsed_log:
            return 0.0

        steps = np.array([entry["step"] for entry in parsed_log])
        rewards = np.array([entry["mean_reward"] for entry in parsed_log])
        group_rewards = np.array([entry["mean_group_reward"] for entry in parsed_log])

        # Compute reward trend (slope)
        trend = np.polyfit(steps, rewards, 1)[0] if len(rewards) > 1 else 0

        # Compute final mean reward
        final_reward = rewards[-1] if rewards.size > 0 else 0

        # Compute variance in rewards (penalize instability)
        reward_variance = np.var(rewards) if rewards.size > 1 else 0

        # Compute mean group reward trend
        group_trend = np.polyfit(steps, group_rewards, 1)[0] if len(group_rewards) > 1 else 0

        # Compute final heuristic score (parameter-free approach)
        score = trend + final_reward - reward_variance + group_trend
        return score

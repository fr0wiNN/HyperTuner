# TODO: Change file comment header
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
import math
import os.path
import random
import sys
import yaml


def load_yaml(file_path: str) -> dict:
    """
    Load tuner YAML config from a given file path.

    Args:
        file_path (str): Path to the YAML config file.

    Returns:
          dict: Parsed tuner YAML config.

    Raises:
        FileNotFoundError: If file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file)
            return data

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    except yaml.YAMLError as e:
        print(f"Error: Failed to pase YAML file. Details: {e}")
        sys.exit(1)


class ConfigManager:

    def __init__(self, conf: dict) -> None:
        """
        Constructs config_generator object and sets abstract rules for config population generation.

        Args:
            conf (dict): The dictionary containing tuner YAML config.
        """
        # Retrieve algorithm rules
        algorithm_rules = conf['algorithm_rules']

        ## Random seed for reproducibility
        self.random_seed = algorithm_rules['random_seed']
        random.seed(self.random_seed)

        ## Starting steps for the population
        self.starting_steps = algorithm_rules['starting_steps']

        ## Starting population size
        self.population_size = algorithm_rules['population_size']

        ## Steps increase for updating max steps
        self.steps_increase = algorithm_rules['steps_increase']

        # Randomization rules for hyperparameters
        self.randomization_rules = conf['randomization_rules']

        self.training_template = load_yaml("tuner_config/training_template/training_template_config.yaml")

    def _apply_randomization(self, config: dict) -> dict:
        """
        Applies randomization rules to generate a new configuration.

        Args:
            config (dict): The base training configuration.

        Returns:
            dict: The randomized training configuration.
        """
        new_config = config.copy()

        for category, params in self.randomization_rules.items():
            # Navigate to the correct section in the template
            target_section = new_config["behaviors"]["SoccerTwos"]

            if category not in target_section:
                raise ValueError(f"Unknown category in randomization_rules: {category}")

            for param, rule in params.items():

                if param not in target_section[category]:
                    raise ValueError(f"Unknown parameter '{param}' in category '{category}'")

                self._apply_rule(target_section[category], param, rule, f"{category}.{param}")

            # TODO: I see a possibility of using this method for updating max steps, when sample is "promoting" - it goes to next generation
            # Special case: Override `max_steps` with the starting_steps
            target_section["max_steps"] = self.starting_steps

        return new_config

    def _apply_rule(self, section: dict, param: str, rule: dict, param_path: str) -> None:
        """
        Applies a randomization rule to a given parameter.

        Args:
            section (dict): The section of the config where the parameter is located.
            param (str): The name of the parameter to modify.
            rule (dict): The rule dict containing the randomization details.
            param_path (str): The full parameter path for debugging.
        """
        # If rule is a nested dictionary (like reward_signals -> extrinsic), go deeper
        if isinstance(rule, dict) and "type" not in rule:
            for sub_param, sub_rule in rule.items():
                if not isinstance(sub_rule, dict) or "type" not in sub_rule:
                    raise KeyError(f"Missing 'type' in nested parameter '{sub_param}' under '{param_path}'")

                self._apply_rule(section[param], sub_param, sub_rule, f"{param_path}.{sub_param}")
            return  # Stop further processing as we've handled the nested case

        # Ensure the rule has a valid 'type'
        if not isinstance(rule, dict) or "type" not in rule:
            raise KeyError(f"Missing 'type' in parameter '{param}' under '{param_path}'")

        param_type = rule["type"]

        # Apply randomization based on type
        if param_type == "static":
            section[param] = rule["value"]
        elif param_type == "uniform":
            section[param] = random.uniform(rule["min"], rule["max"])
        elif param_type == "log":
            section[param] = 10 ** random.uniform(
                math.log10(rule["min"]), math.log10(rule["max"])
            )
        elif param_type == "discrete":
            section[param] = random.choice(rule["choices"])
        elif param_type == "doubling":
            min_exp = int(math.log2(rule["min"]))
            max_exp = int(math.log2(rule["max"]))
            section[param] = 2 ** random.randint(min_exp, max_exp)
        elif param_type == "default":
            pass
        else:
            raise ValueError(f"Unknown randomization type: {param_type}")

    def generate_starting_population(self, output_dir: str) -> None:
        """
        Creates a population of training configs based of defined tuning config...

        Args:
            output_dir (str): Directory to save the population.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for i in range(self.population_size):
            config_name = f"config_sample{i}.yaml"
            new_config = self._apply_randomization(self.training_template)

            with open(os.path.join(output_dir, config_name), "w", encoding="utf-8") as file:
                yaml.dump(new_config, file, default_flow_style=False)

            print(f"Generated: {config_name}")


    def promote(self, config_file: str) -> None:
        """
        Promotes a configuration file by doubling its 'max_steps'.

        Args:
            config_file (str): The path to the configuration file.
        """
        try:
            with open(config_file, 'r', encoding='utf-8') as file:
                config_data = yaml.safe_load(file)

            target_section = config_data["behaviors"]["SoccerTwos"]
            if "max_steps" in target_section:
                target_section["max_steps"] *= 2
            else:
                raise KeyError("max_steps not found in configuration file")

            with open(config_file, 'w', encoding='utf-8') as file:
                yaml.dump(config_data, file, default_flow_style=False)

            print(f"Promoted: {config_file} (max_steps doubled)")
        except (FileNotFoundError, KeyError, yaml.YAMLError) as e:
            print(f"Error promoting config {config_file}: {e}")




import os
import subprocess
from config_manager import ConfigManager
from log_parser import LogParser
from heuristic_calculator import HeuristicCalculator

class TrainingManager:
    def __init__(self, conf: dict):
        self.conf = conf
        self.working_directory = "experiments"
        self.experiment_number = -1
        self.current_population = {}

        # Possible to declare those classes with some optional configuration
        # e.g. log_parser with set location of GPU/CPU logs
        # e.g. heuristic_calculator with preset version of heuristics being used
        self.config_manager = ConfigManager(self.conf)
        self.log_parser = LogParser()
        self.heuristic_calculator = HeuristicCalculator()

    def find_best_config(self):
        self._setup()
        self._run_SHA()

    def _setup(self):
        print("Setting up training...")
        self.config_manager.generate_starting_population(os.path.join(self.working_directory, f"experiment_{self.experiment_number}", "population"))

        # Initialize population with correct order
        config_files = sorted(
            [f for f in os.listdir(os.path.join(self.working_directory, f"experiment_{self.experiment_number}", "population")) if f.endswith(".yaml")],
            key=lambda x: int(x.replace(".yaml", "").split('_')[-1][6:])  # Extract only the numeric part
        )

        # Set heuristics to zero
        self.current_population = {config: 0 for config in config_files}

    def _run_SHA(self):

        gen_counter = 1
        # Main SHA loop
        while len(self.current_population) > 1:
            print(f"--- Running SHA {gen_counter} generation with {len(self.current_population)} candidates ---")

            # Evaluate each configuration sample
            for config_file in self.current_population.keys():
                training_config_path = os.path.join(self.working_directory, f"experiment_{self.experiment_number}", "population", config_file)

                # Construct command
                command = [
                    "mlagents-learn",
                    training_config_path,
                    f"--run-id=experiment{self.experiment_number}_gen{gen_counter}_{config_file.split('.')[0]}",
                    f"--seed={self.conf['algorithm_rules']['random_seed']}",
                    "--no-graphics",
                    "--env=../ml-agents-fix-numpy-release-21-branch/builds/UnityEnvironment.exe",
                    f"--results-dir={os.path.join(self.working_directory, f'experiment_{self.experiment_number}', 'results')}"
                ]

                # Run ML-Agents training
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )

                # Capture log output
                log_output = ""
                for line in process.stdout:
                    log_output += line + "\n"

                # Wait for process to complete
                process.wait()

                # Calculate heuristic
                parsed_log = self.log_parser.parse(log_output)
                heuristic_value = self.heuristic_calculator.calculate(parsed_log)
                self.current_population[config_file] = heuristic_value
                print(f"{config_file} scored {heuristic_value}")

            # Sort by heuristic value and retain the top half
            sorted_population = sorted(self.current_population.items(), key=lambda x: x[1], reverse=True)
            self.current_population = dict(sorted_population[:len(sorted_population) // 2])

            # Promote half of the best samples
            for config_file in self.current_population.keys():
                self.config_manager.promote(os.path.join(self.working_directory, f"experiment_{self.experiment_number}", "population" , config_file))
                self.current_population[config_file] = 0  # Reset heuristic score

            gen_counter += 1

        # Print winner configuration
        best_config = list(self.current_population.keys())[0]
        print(f"Best configuration found: {best_config}")
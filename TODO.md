# Hyper Tuner TODO

## Tasks

### Development
- [ ] Create git repository
- [ ] Resolve Unity Profiler issue of GUI access only

### After-Development
- [ ] Ensure that the usage of **Hyper Tuner** is easy and that it makes reproducible experiments
- [ ] Enable smooth transfer to ml-agents virtual environment. The `git clone` step in venv is also an option - if so, ensure, there are no path problems issues - smooth program transfer
- [ ] Prepare solid config_manual.md that explains how to write **Hyper Tuner** configuration file together with all implemented methods for params randomization.
- [ ] Prepare -h flag for hyper_tuner.py, which should give some basic info about the usage.
- [ ] Refactor project structure - Ensure solid folder/files structure, that has the most sense

### Right before Delivery
- [ ] Prepare solid documentation on how to use **Hyper Tuner**
- [ ] Check the documentation if it still applies
- [ ] Add licencing

## Experiments and Testing

- [ ] Test if random.seed() works as intended. Seed functionality, was made for experiments reproducibility. See if created config files are the same for a multiple runs with the same random_seed in tuner_config.yaml

## Current Classes

### config_manager.py

- [ ] Spend more time for developing more suitable way of working with nested categories - use abstraction separation on _apply_randomization() which currently handles the iteration of training template  
- [ ] _apply_randomization() needs to be readable
- [ ] _apply_randomization() has a possibility for being more abstract - abstract _replace() function, which could be used, both for replacing with random rules **and** for single replacement of max_steps

### heuristic_calculator.py

### log_parser.py

### hyper_tuner.py

### training_manager.py

- [ ] change dummy `cat` command to actual `train-agents`

## Classes to Implement

- [x] training_manager.py - would connect training execution together with heuristic calculator. I could also manage the elimination and promotion process. We need to separate the algorithm flow from hyper_tuner.py, which should handle CLI interactions - optional flags, input configs and so on...
"""
File with all required parameters.
Note the required parameters for TFSearch - See TFSearch.py
"""
import numpy as np

cns_size = 10
step_size = 0.1
obs_size = 17
action_size = 7
task_name = "HalfCheetah-v1"
EVOL_MODE = True
## Parameters dict - Could possibly split into multiple dicts - env+agent+evol
EVOL_P = {  # task-specific params
    "obs_size": obs_size,
    "max_num_episodes": 3,
    "evaluation_duration": 500,
    # CTRNN params
    "cns_size": cns_size,
    "step_size": step_size,
    "out_size": action_size,
    # required params for TFSearch
    "genotype_size": cns_size ** 2
    + 2 * cns_size
    + obs_size * cns_size
    + cns_size * action_size,
    "genotype_min_val": 0,
    "genotype_max_val": 1,
    "scaling_high": np.hstack(
        [
            [2] * cns_size,
            [5] * cns_size,
            [5] * cns_size ** 2,
            [5] * ((obs_size * cns_size) + (cns_size * action_size)),
        ]
    ),
    "scaling_low": np.hstack(
        [
            [1] * cns_size,
            [-5] * cns_size,
            [-5] * cns_size ** 2,
            [-5] * ((obs_size * cns_size) + (cns_size * action_size)),
        ]
    ),
    "pop_size": 100,
    "max_gens": 500,
    "elitist_fraction": 0.4,
    "mutation_variance": 10,
    "crossover_fraction": 0.1,
    "crossover_probability": 0.3,
    "device": "/cpu:0",
}

if __name__ == "__main__":
    print("Printing all params...")
    import pprint

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(EVOL_P)

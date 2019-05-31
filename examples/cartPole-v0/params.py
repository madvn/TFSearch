"""
File with all required parameters.
Note the required parameters for TFSearch - See TFSearch.py
"""
import numpy as np

cns_size = 2
step_size = 1.0
obs_size = 4
action_size = 2
task_name = "CartPole-v0"
## Parameters dict - Could possibly split into multiple dicts for better readability
EVOL_P = {  # task-specific params
    "max_num_episodes": 5,
    "max_episode_len": 5000,
    "obs_size": obs_size,
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
    "max_gens": 100,
    "elitist_fraction": 0.1,
    "mutation_variance": 0.1,
    "crossover_fraction": 0.1,
    "crossover_probability": 0.3,
    "device": "/cpu:0",
}

if __name__ == "__main__":
    print("Printing all params...")
    import pprint

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(EVOL_P)

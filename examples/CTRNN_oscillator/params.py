"""
File with all required parameters.
Note the required parameters for TFSearch - See TFSearch.py
"""
import numpy as np

cns_size = 5
step_size = 0.01

## Parameters dict - Could possibly split into multiple dicts for better readability
EVOL_P = {
    "cns_size": cns_size,
    "step_size": step_size,
    "transient_duration": 1 / step_size,
    "eval_duration": 10 / step_size,
    "genotype_size": cns_size ** 2 + 2 * cns_size,
    "genotype_min_val": 0,
    "genotype_max_val": 1,
    "scaling_high": np.hstack(
        [[2] * cns_size, [5] * cns_size, [5] * cns_size ** 2]
    ),  # upper bound for scaling genotype - [taus,biases,weights]
    "scaling_low": np.hstack(
        [[1] * cns_size, [-5] * cns_size, [-5] * cns_size ** 2]
    ),  # lower bound for scaling genotype - [taus,biases,weights]
    "pop_size": 100,
    "max_gens": 100,
    "elitist_fraction": 0.1,
    "mutation_variance": 0.1,
    "crossover_fraction": 0.1,
    "crossover_probability": 0.3,
    "num_evol_runs": 5,
}

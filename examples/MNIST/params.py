"""
File with all required parameters.
Note the required parameters for TFSearch - See TFSearch.py
"""
import numpy as np

## Required Parameters
EVOL_P = {
    "pop_size": 50,
    "genotype_size": 7850,
    "genotype_min_val": 0,
    "genotype_max_val": 1,
    "scalingUB": np.ones(7850) * 50,  # upper bound for scaling genotype
    "scalingLB": np.zeros(7850),  # lower bound for scaling genotype
    "popSize": 100,
    "maxGens": 2000,
    "elitist_fraction": 0.1,
    "mutation_variance": 0.05,
    "numSamples": 20000,
    "numTrials": 1,
}

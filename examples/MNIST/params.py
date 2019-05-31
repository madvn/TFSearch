"""
File with all required parameters.
Note the required parameters for TFSearch - See TFSearch.py
"""
import numpy as np

## Required Parameters
EVOL_P = {
    "genotypeSize": 7850,
    "genotypeMinVal": 0,
    "genotypeMaxVal": 1,
    "scalingUB": np.ones(7850) * 50,  # upper bound for scaling genotype
    "scalingLB": np.zeros(7850),  # lower bound for scaling genotype
    "popSize": 100,
    "maxGens": 2000,
    "elitistFraction": 0.1,
    "mutationVariance": 0.05,
    "numSamples": 20000,
    "numTrials": 1,
}

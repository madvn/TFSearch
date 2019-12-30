"""
A sample file showing an example of how to use TFSearch for setting up an
evolutionary search and running it.
The goal here is to evolve an image to match the target image - see targetImage.bmp
This main script defines the genotype-phenotype conversion and fitness evaluation
and uses TFSearch to step through generations
Author: Madhavun Candadai
Date Created: Sep 18, 2017
"""

import numpy as np
import tensorflow as tf
import scipy.misc
from TFSearch import *

## Required Parameters
EVOL_P = {
    "genotype_size": 256,
    "genotype_min_val": 0,
    "genotype_max_val": 1,
    "scaling_high": np.ones(256),
    "scaling_low": np.zeros(256),
    "pop_size": 200,
    "max_gens": 7000,
    "elitist_fraction": 0.4,
    "mutation_variance": 0.1,
    "crossover_fraction": 0.1,
    "crossover_probability": 0.3,
    "num_evol_runs": 5,
}

# Tensorflow placeholder for target image
target_pl = tf.placeholder(tf.float32)

# genotype-phenotype conversion as a Tensorflow op
with tf.name_scope("genotype_to_phenotype"):
    phenotypes = tf.round(pop_pl)

# Fitness estimating as a proportion of matching pixels as Tensorflow op
with tf.name_scope("fitness_evaluation"):
    fitness = (
        tf.reduce_sum(
            tf.cast(
                tf.equal(tf.tile(target_pl, [EVOL_P["pop_size"], 1]), phenotypes),
                tf.float32,
            ),
            axis=1,
        )
        / EVOL_P["genotype_size"]
    )

# Load target image
tgt = np.loadtxt("./targetImage.dat")
tgt = np.reshape(tgt, [1, EVOL_P["genotype_size"]])

# initialize a random population
this_pop = np.random.uniform(
    size=(EVOL_P["pop_size"], EVOL_P["genotype_size"]),
    low=EVOL_P["genotype_min_val"],
    high=EVOL_P["genotype_max_val"],
)

# create TensorFlow session
sess = tf.Session()

# creating the TFSearch graph
tfs = TFSearch(EVOL_P)

print("\nStarting to run session")
print(["Generation", "Best Fitnesss", "Mean Fitness"])
for gen in range(EVOL_P["max_gens"]):
    # Given a population, first compute fitness
    fs = sess.run(fitness, {pop_pl: this_pop, target_pl: tgt})

    # printing out some stats occasionally
    if gen == 0 or (gen + 1) % 1000 == 0:
        bestFs = np.max(fs)
        meanFs = np.mean(fs)
        print([gen + 1, bestFs, meanFs])

    # Producing a new population given this population and its fitnesss using Generation_Reproduce
    this_pop = sess.run(tfs.step_generation, {pop_pl: this_pop, fits_pl: fs})

# Since the population is always ordered, the best individual is always the first
bestIndividual = [this_pop[0][:]]
bestImage = sess.run(phenotypes, {pop_pl: bestIndividual})
bestImage = list(map(int, bestImage[0]))
bestImage = np.transpose(np.reshape(bestImage, [16, 16]))
# Saving best individual as image
scipy.misc.imsave("./bestEvolvedImage.bmp", bestImage)

"""
Evolutionary optimization using Tensorflow
This file implements the class TFSearch which takes a population and fitnesses,
and returns a new population effectively taking one generation step
Author: Madhavun Candadai
Date Created: Sep 17, 2017
Decorators from: https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
"""

import functools
import numpy as np
import tensorflow as tf


def doublewrap(function):
    """
    A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)

    return decorator


@doublewrap
def define_scope(function, scope=None, *args, **kwargs):
    """
    A decorator for functions that define TensorFlow operations. The wrapped
    function will only be executed once. Subsequent calls to it will directly
    return the result so that operations are added to the graph only once.
    The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    attribute = "_cache_" + function.__name__
    name = scope or function.__name__

    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(name, *args, **kwargs):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator


# Global placeholders for population and fitness values
pop_pl = tf.placeholder(tf.float32, shape=[None, None])
fits_pl = tf.placeholder(tf.float32, shape=[None])


class TFSearch:
    def __enter__(self):
        return self

    def __init__(self, EVOL_P):
        """
            The TF graph is defined here. Attributes are defined using
            decorators so they can be used like functions without growing the
            graph each time they are called; and also to define scopes within
            the graph for better readability.

            Args:
            EVOL_P - a dictionary of evolutionary params with required keys
                'pop_size' = scalar: population size
                'mutation_variance' = scalar: variance for 0 mean mutation gaussian
                'elitist_fraction' = scalar: proportion of best fitness individuals
                                    kept as is in new population
                'genotype_size' = scalar: dimensionality of genotype
                'genotype_min_val' = scalar: lower bound of genotype values
                'genotype_max_val' = scalar: upper bound of genotype values

            Output:
            Graph compute returns newPop: shape=(EVOL_P['pop_size'],EVOL_P['genotype_size'])
        """
        self.pop_size = EVOL_P["pop_size"]
        self.genotype_size = EVOL_P["genotype_size"]
        self.mutation_variance = EVOL_P["mutation_variance"]
        self.genotype_min_val = EVOL_P["genotype_min_val"]
        self.genotype_max_val = EVOL_P["genotype_max_val"]
        self.elitist_fraction = EVOL_P["elitist_fraction"]
        self.crossover_fraction = EVOL_P["crossover_fraction"]
        self.crossover_probability = EVOL_P["crossover_probability"]
        self.pop = pop_pl
        self.fitness = fits_pl
        self.elitist_selection
        self.crossover
        self.mutation
        self.step_generation

    @define_scope
    def elitist_selection(self):
        # sort fitness and rank the pop
        _, indices = tf.nn.top_k(self.fitness, k=self.pop_size, sorted=True)
        ranks = tf.cast(indices, tf.int32)

        # identify parents using elitist_fraction
        num_parents = int(np.floor(self.elitist_fraction * self.pop_size))
        self.parents = tf.gather(self.pop, tf.slice(ranks, [0], [num_parents]))

        # make copies of parents for new pop
        self.num_children = self.pop_size - num_parents
        num_parent_reps = int(np.ceil(self.num_children / float(num_parents)))
        self.children = tf.tile(self.parents, [num_parent_reps, 1])
        return [self.parents, self.children]

    @define_scope
    def crossover(self):
        # children for crossover
        num_crossover_children = int(
            np.floor(self.num_children * self.crossover_fraction)
        )
        crossover_children, noncross_children = tf.split(
            self.children, [num_crossover_children, -1], axis=0
        )

        # shuffle the children and produce a subset from that
        shuffled_children = tf.transpose(
            tf.random_shuffle(tf.transpose(tf.random_shuffle(crossover_children)))
        )
        cprobs = tf.random_uniform(
            shape=tf.shape(crossover_children), minval=0, maxval=1
        )
        crossover_points = tf.less(
            cprobs, self.crossover_probability
        )  # tf.where(tf.less(cprobs,0.5),tf.ones(tf.shape(dat)),tf.zeros(tf.shape(dat)))
        crossover_points_f = tf.cast(crossover_points, tf.float32)
        cped1 = tf.multiply(shuffled_children, crossover_points_f)

        # identify unchanged positions that don't crossover
        noncrossover_points = tf.logical_not(crossover_points)
        noncrossover_points_f = tf.cast(noncrossover_points, tf.float32)
        cped2 = tf.multiply(crossover_children, noncrossover_points_f)

        # combine shuffled subset with original children
        crossed_children = tf.add(cped1, cped2)

        # combine with crossed children with children that weren't selected for crossover
        self.crossed_children_to_mutate = tf.concat(
            [crossed_children, noncross_children], 0
        )

        return self.crossed_children_to_mutate

    @define_scope
    def mutation(self):
        # mutate the kinds using mutation_variance
        self.crossed_mutant_children = (
            self.crossed_children_to_mutate
            + tf.random_normal(
                shape=tf.shape(self.children), mean=0.0, stddev=self.mutation_variance
            )
        )
        return self.crossed_mutant_children

    @define_scope
    def step_generation(self):
        # Concat self.parents and mutated kids to form new pop - this might be larger than required
        new_pop = tf.concat([self.parents, self.crossed_mutant_children], 0)
        # self.new_pop_s = tf.shape(new_pop)

        # Make sure pop_size remains the same
        new_eqiv_pop = tf.slice(new_pop, [0, 0], [self.pop_size, self.genotype_size])
        # self.new_eqiv_pop_s = tf.shape(new_eqiv_pop)

        # Bound pop so the genotpyes remain in the required range
        new_bounded_pop = tf.clip_by_value(
            new_eqiv_pop, self.genotype_min_val, self.genotype_max_val
        )
        # self.new_bounded_pop_s = tf.shape(new_bounded_pop)
        return new_bounded_pop

    def __exit__(self, *err):
        pass

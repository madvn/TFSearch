"""
A sample file that implements the problem-specific part of the evolutionary
optimization: genotype-phenotype conversion and fitness evaluation.
This example is for evolving weights and bias for MNIST classification
This is written using tensorflow but it does not necessarily have to be.
NetworkPop implements a fully vectorized implementation of working on the
whole population as single matrix
The same template as TFSearch.py has been used to show how Tensorflow
can be used here.
Author: Madhavun Candadai
Date Created: Sep 18, 2017
Decorators from: https://gist.github.com/danijar/8663d3bbfd586bffecf6a0094cd116f2
"""

import functools
import numpy as np
import tensorflow as tf
from TFSearch import *


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


ins = tf.placeholder(tf.float32)
ys = tf.placeholder(tf.int64)


class NetworkPop:
    def __enter__(self):
        # print 'Creating graph'
        return self

    def __init__(self, EVOL_P):
        """
        This class defines the graph for converting a given genotype matrix to a
        phenotype matrix, and for estimating the fitnesses of the phenotypes.
        This implementation could be using TensorFlow (preferably) or could use
        numpy.
        """
        self.setParams(EVOL_P)
        self.pop = pop_pl
        self.inputs = ins
        self.y = ys
        self.numSamples = EVOL_P["numSamples"]  # ns
        self.testDataFlag = 0
        self.phenotypes
        self.fitness

    @define_scope
    def phenotypes(self):
        # first scale them to be weights and biases
        phens = self.pop * (self.scalingUB - self.scalingLB) + self.scalingLB

        # extract weight and bias values from phenotypes
        popWeights = tf.slice(phens, [0, 0], [self.popSize, 7840])
        popBiases = tf.slice(phens, [0, 7840], [self.popSize, 10])

        # now reshape to be [matrix of [matrics for each network in the population]]
        self.weightMatrices = tf.squeeze(
            tf.transpose(
                tf.reshape(popWeights, [1, 10 * self.popSize, 784]), perm=[0, 2, 1]
            )
        )
        self.weightMatrices_s = tf.shape(self.weightMatrices)
        self.biasMatrices = tf.tile(
            tf.reshape(popBiases, [1, 10 * self.popSize]), [self.numSamples, 1]
        )
        self.biasMatrices_s = tf.shape(self.biasMatrices)

        return [self.weightMatrices, self.biasMatrices]

    @define_scope
    def fitness(self):
        # fits = tf.reduce_sum(tf.cast(tf.equal(tf.tile(self.targetImg,[self.popSize,1]),self.phenotypes),tf.float32),axis=1)/self.genotypeSize
        # getting data to test
        labels = tf.squeeze(tf.tile(self.y, [1, self.popSize]))

        # compute weighted sum + bias
        popYlogits = tf.matmul(self.inputs, self.weightMatrices) + self.biasMatrices

        # should convert back to proper dimensions to estimate softmax and consequently the estimated label correctly
        ylogits = tf.concat(tf.split(popYlogits, [10] * self.popSize, axis=1), axis=0)
        self.ylogits_s = tf.shape(ylogits)
        yhats = tf.nn.softmax(ylogits)
        labelEstimates = tf.argmax(yhats, axis=1)
        self.labelEstimates_s = tf.shape(labelEstimates)
        self.labels_s = tf.shape(labels)

        popPerfs = tf.expand_dims(
            tf.cast(tf.equal(labels, labelEstimates), tf.float32), 0
        )

        # split perfs per individual and compute mean
        fits = tf.reduce_mean(
            tf.concat(
                tf.split(popPerfs, [self.numSamples] * self.popSize, axis=1), axis=0
            ),
            axis=1,
        )
        return fits

    def setParams(self, E_P):
        self.popSize = E_P["pop_size"]
        self.genotypeSize = E_P["genotype_size"]
        self.scalingUB = E_P["scalingUB"]
        self.scalingLB = E_P["scalingLB"]

    def __exit__(self, *err):
        # print 'Deleting graph'
        pass

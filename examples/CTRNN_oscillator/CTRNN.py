"""
This CTRNN_popSim class implements a fully-vectorized Tensorflow implementation of
1. genotype-phenotype conversion
2. euler step the CTRNN
with the whole population of networks data-typed as a single matrix.
Author: Madhavun Candadai
Date Created: Sep 23, 2017
"""
import functools
import numpy as np
import tensorflow as tf
from TFSearch import *  # just to use the popPl placeholder


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


# genotype to phenotype conversion
# class since we could call with different size pops using different instances
class MakeCTRNN(object):
    def __enter__(self):
        return self

    def __init__(self, EVOL_P):
        self.genotype_size = EVOL_P["genotype_size"]
        self.cns_size = EVOL_P["cns_size"]
        self.pop_size = EVOL_P["pop_size"]
        self.scaling_high = EVOL_P["scaling_high"]
        self.scaling_low = EVOL_P["scaling_low"]
        self.step_size = EVOL_P["step_size"]
        self.pop = pop_pl
        self.genotype_to_phenotype

    @define_scope
    def genotype_to_phenotype(self):
        # first scale weights, taus and biases
        phens = self.pop * (self.scaling_high - self.scaling_low) + self.scaling_low

        # split phens into its components - each component has the values for the whole pop
        pop_taus, pop_biases, pop_weights = tf.split(
            phens, [self.cns_size, self.cns_size, self.cns_size ** 2], axis=1
        )  # taus, biases and weights

        # reshape them as required into 1-D vectors but 1X(cns_size*pop_size)
        self.tau_matrices = tf.reshape(pop_taus, [1, self.cns_size * self.pop_size])
        self.bias_matrices = tf.reshape(pop_biases, [1, self.cns_size * self.pop_size])
        self.weight_matrices = tf.squeeze(
            tf.transpose(
                tf.reshape(
                    pop_weights, [1, self.cns_size * self.pop_size, self.cns_size]
                ),
                perm=[0, 2, 1],
            )
        )
        self.weight_matrices_s = tf.shape(self.weight_matrices)

        return [self.tau_matrices, self.bias_matrices, self.weight_matrices]

    def __exit__(self, *err):
        pass


state_pl = tf.placeholder(tf.float32, shape=(None, None))
output_pl = tf.placeholder(tf.float32, shape=(None, None))


class CTRNN:
    def __enter__(self):
        return self

    def __init__(self, EVOL_P, weights_m, biases_m, taus_m):
        """
        This CTRNN class instantiates the computational graph for the network

        Args:
        EVOL_P - a dict with required params
            'genotype_size' = scalar: dimensionality of genotype,
            'cns_size' = scalar: number of neurons in the network,
            'pop_size' = scalar: number of individuals,
            'scaling_high' = vector (1 X genotype_size): upper bound to scale genotype,
            'scaling_low' = vector (1 X genotype_size): lower bound to scale genotype,
            'step_size' = scalar: euler integration step size
        state_pl - placeholder for state at previous time step
        output_pl - placeholder for output at previous time step

        Output:
        Graph compute of
            'step_state' returns state at next time step
            'step_output' returns output at next time step
        """
        self.genotype_size = EVOL_P["genotype_size"]
        self.cns_size = EVOL_P["cns_size"]
        self.pop_size = EVOL_P["pop_size"]
        self.scaling_high = EVOL_P["scaling_high"]
        self.scaling_low = EVOL_P["scaling_low"]
        self.step_size = EVOL_P["step_size"]
        self.pop = pop_pl
        self.state_p = state_pl  # past state
        self.output_p = output_pl  # past output
        self.weights_T = tf.cast(weights_m, tf.float32)
        self.biases_T = tf.cast(biases_m, tf.float32)
        self.taus_T = tf.cast(taus_m, tf.float32)
        self.step_state
        self.step_output

    @define_scope
    def step_state(self):
        # computing the totalinput
        tiled_past_outputs = tf.tile(
            self.output_p, [self.cns_size, 1]
        )  # this is now cns_size X (cns_size*pop_size)
        spot_inputs = tf.multiply(
            tiled_past_outputs, self.weights_T
        )  # this now has, at each spot, one weighted input line
        rearr_spot_inputs = tf.concat(
            tf.split(spot_inputs, [self.cns_size] * self.pop_size, axis=1), axis=0
        )  # rearranged spot_inputs to get totalinput
        inputs = tf.expand_dims(
            tf.reduce_sum(rearr_spot_inputs, axis=1), axis=0
        )  # this is now 1 X (cns_size*pop_size)

        # updating states
        pop_states = self.state_p + self.step_size * (
            tf.multiply(-self.state_p + inputs, 1 / self.taus_T)
        )
        return pop_states

    @define_scope
    def step_output(self):
        # compute output for whole pop
        pop_outputs = tf.sigmoid(self.step_state + self.biases_T)
        return pop_outputs

    def __exit__(self, *err):
        pass

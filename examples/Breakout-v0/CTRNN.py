"""
This CTRNN_popSim class implements a fully-vectorized Tensorflow implementation of
1. genotype-phenotype conversion
2. euler step the CTRNN
with the whole population of networks data-typed as a single matrix.
Author: Madhavun Candadai
Date Created: Sep 23, 2017
"""
import numpy as np
import tensorflow as tf
from TFSearch import *  # just to use the popPl placeholder


class MakeCTRNN:
    def __enter__(self):
        return self

    def __init__(self, EVOL_P):
        self.genotype_size = EVOL_P["genotype_size"]
        self.cns_size = EVOL_P["cns_size"]
        self.pop_size = EVOL_P["pop_size"]
        self.scaling_high = EVOL_P["scaling_high"]
        self.scaling_low = EVOL_P["scaling_low"]
        self.step_size = EVOL_P["step_size"]
        self.obs_size = EVOL_P["obs_size"]
        self.pop_pl = pop_pl

        with tf.name_scope("genotype_to_phenotype"):
            phens = (
                self.pop_pl * (self.scaling_high - self.scaling_low) + self.scaling_low
            )

            # split phens into its components - each component has the values for the whole pop
            pop_taus, pop_biases, pop_weights, in_weights = tf.split(
                phens,
                [
                    self.cns_size,
                    self.cns_size,
                    self.cns_size ** 2,
                    self.obs_size * self.cns_size,
                ],
                axis=1,
            )  # taus, biases and weights

            # reshape them as required into 1-D vectors but 1X(cns_size*pop_size)
            self.tau_matrices = tf.reshape(pop_taus, [1, self.cns_size * self.pop_size])
            self.bias_matrices = tf.reshape(
                pop_biases, [1, self.cns_size * self.pop_size]
            )

            self.weight_matrices = tf.squeeze(
                tf.transpose(
                    tf.reshape(
                        pop_weights, [1, self.cns_size * self.pop_size, self.cns_size]
                    ),
                    perm=[0, 2, 1],
                )
            )
            self.weight_matrices_s = tf.shape(self.weight_matrices)

            self.in_weight_matrices = tf.squeeze(
                tf.transpose(
                    tf.reshape(
                        in_weights, [1, (self.obs_size * self.pop_size), self.cns_size]
                    ),
                    perm=[0, 2, 1],
                )
            )

    def __exit__(self, *err):
        pass


state_pl = tf.placeholder(tf.float32, shape=(None, None), name="state_t-1")
output_pl = tf.placeholder(tf.float32, shape=(None, None), name="output_t-1")
input_pl = tf.placeholder(tf.float32, shape=(None, None), name="input_t")


class CTRNN:
    def __enter__(self):
        return self

    def __init__(self, EVOL_P):
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
        self.obs_size = EVOL_P["obs_size"]
        self.state_pl = state_pl  # past state
        self.output_pl = output_pl  # past output
        self.input_pl = input_pl  # current input

        with tf.device("/cpu:0"):
            self.weights_T = tf.get_variable(
                "weights_T",
                dtype=tf.float32,
                shape=(self.cns_size, self.cns_size * self.pop_size),
            )  # tf.cast(weights_m,tf.float32)
            self.biases_T = tf.get_variable(
                "biases_T", dtype=tf.float32, shape=(1, self.cns_size * self.pop_size)
            )  # tf.cast(biases_m,tf.float32)
            self.taus_T = tf.get_variable(
                "taus_T", dtype=tf.float32, shape=(1, self.cns_size * self.pop_size)
            )  # tf.cast(taus_m,tf.float32)
            self.in_weights = tf.get_variable(
                "in_weights",
                dtype=tf.float32,
                shape=(self.cns_size, self.obs_size * self.pop_size),
            )

        with tf.name_scope("step_state"):
            # computing the total input from other neurons
            tiled_past_outputs = tf.tile(
                self.output_pl, [self.cns_size, 1]
            )  # this is now cns_size X (cns_size*pop_size)
            spot_inputs = tf.multiply(
                tiled_past_outputs, self.weights_T
            )  # this now has, at each spot, one weighted input line
            rearr_spot_inputs = tf.concat(
                tf.split(spot_inputs, [self.cns_size] * self.pop_size, axis=1), axis=0
            )  # rearranged spot_inputs
            inputs = tf.expand_dims(
                tf.reduce_sum(rearr_spot_inputs, axis=1), axis=0
            )  # this is now 1 X (cns_size*pop_size)

            # computing the total external inputs
            tiled_ex_inputs = tf.tile(self.input_pl, [self.cns_size, 1])
            spot_ex_inputs = tf.multiply(tiled_ex_inputs, self.in_weights)
            rearr_spot_ex_inputs = tf.concat(
                tf.split(spot_ex_inputs, [self.obs_size] * self.pop_size, axis=1),
                axis=0,
            )
            external_inputs = tf.expand_dims(
                tf.reduce_sum(rearr_spot_ex_inputs, axis=1), axis=0
            )

            # updating states
            self.step_state = self.state_pl + self.step_size * (
                tf.multiply(-self.state_pl + inputs + external_inputs, 1 / self.taus_T)
            )

        with tf.name_scope("step_output"):
            # compute output for whole pop
            self.step_output = tf.sigmoid(self.step_state + self.biases_T)

    def __exit__(self, *err):
        pass

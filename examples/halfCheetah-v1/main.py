"""
Evolving CTRNNs for openAI Gym's Ping-v0
Author - Madhavun Candadai
Date Created - Oct 9, 2017
"""
import gym
import tensorflow as tf
import numpy as np
import sys
from TFSearch import *
from CTRNN import *
from params import *


def evol_ctrnn_pong(genotype_to_sim=None):
    if EVOL_MODE:
        # Initializing writer to collect stats
        writer = tf.summary.FileWriter("./logs/")

        # Initializing poriton of the graph for generation stepping
        tfs = TFSearch(EVOL_P)
    else:
        with tf.variable_scope("population", reuse=None):
            pop = tf.get_variable(
                "pop", shape=(EVOL_P["pop_size"], EVOL_P["genotype_size"])
            )

    # with tf.variable_scope('phenotypes',reuse=True):
    # Initializing poriton of the graph for CTRNN stepping for the whole population
    ctrnns = CTRNN(EVOL_P)

    # Initializing the environments
    print("Initializing the environments...")
    envs = []
    for _ in range(EVOL_P["pop_size"]):
        envs.append(gym.make(task_name))

    # Running TFSearch
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        tf.global_variables_initializer().run()
        if not EVOL_MODE:
            print(np.shape([genotype_to_sim]), EVOL_P["genotype_size"])
            with tf.variable_scope("population", reuse=True):
                sess.run(
                    tf.assign(
                        tf.get_variable("pop"),
                        tf.convert_to_tensor(
                            np.asarray([genotype_to_sim]), dtype=tf.float32
                        ),
                    )
                )

        # stepping through the generations
        for gen in range(EVOL_P["max_gens"]):
            # print('Generation',gen)
            # Initializing reward vector
            rewards = np.zeros(EVOL_P["pop_size"])

            # make phenotypes and assign to variables so it can be used by CTRNNs class
            sess.run(
                [
                    ctrnns.weight_matrices,
                    ctrnns.bias_matrices,
                    ctrnns.tau_matrices,
                    ctrnns.in_weight_matrices,
                    ctrnns.out_weight_matrices,
                ]
            )

            # Initializing reward vector
            rewards = np.zeros(EVOL_P["pop_size"])

            # simulate a number of episodes to evaluate fitness
            for episode_ind in range(EVOL_P["max_num_episodes"]):
                # print('\tEpisode',episode_ind)
                # Reset task status - Setting all CTRNNs as not done with task
                not_dones = np.ones(EVOL_P["pop_size"])

                # Reset envs - Initializing inputs for the CTRNNs for each episode
                this_inputs = []
                for env in envs:
                    observation = env.reset()
                    this_inputs = np.append(this_inputs, observation)

                # Reset CTRNNs - random initial conditions for CTRNNs
                sess.run(
                    ctrnns.randomize_states,
                    {
                        state_pl: np.random.rand(
                            1, EVOL_P["pop_size"] * EVOL_P["cns_size"]
                        )
                    },
                )
                sess.run(
                    ctrnns.randomize_outputs,
                    {
                        output_pl: np.random.rand(
                            1, EVOL_P["pop_size"] * EVOL_P["cns_size"]
                        )
                    },
                )

                episode_len = 0
                while episode_len < EVOL_P["evaluation_duration"]:
                    # set inputs
                    sess.run(
                        [ctrnns.set_external_inputs],
                        feed_dict={input_pl: [this_inputs]},
                    )
                    # with tf.variable_scope('CTRNN_vars',reuse=True):
                    # sess.run(tf.assign(tf.get_variable('this_inputs'),tf.convert_to_tensor(np.asarray([this_inputs]),dtype=tf.float32)))

                    # Step the network
                    for _ in range(1):  # int(1./step_size)):
                        sess.run([ctrnns.step_state, ctrnns.step_output])
                    # print 'weights_T',np.sum(np.isnan(sess.run(ctrnns.weights_T)))
                    # print 'biases_T',np.sum(np.isnan(sess.run(ctrnns.biases_T)))
                    # print 'taus_T',np.sum(np.isnan(sess.run(ctrnns.taus_T)))
                    # print 'in_weights',np.sum(np.isnan(sess.run(ctrnns.in_weights)))
                    # print 'out_weights',np.sum(np.isnan(sess.run(ctrnns.out_weights)))
                    # print 'states',np.sum(np.isnan(sess.run(ctrnns.state_v)))
                    # print 'outputs',np.sum(np.isnan(sess.run(ctrnns.output_v)))
                    control_outputs = sess.run(ctrnns.control_outputs)
                    # print 'control outputs',np.sum(np.isnan(control_outputs)),'\n\n'

                    # Construct the inputs for next time step by stepping envs
                    this_inputs = np.zeros((EVOL_P["pop_size"], EVOL_P["obs_size"]))
                    for ind, env in enumerate(envs):
                        if not_dones[ind]:
                            if not EVOL_MODE:
                                env.render()
                            observation, reward, done, _ = envs[ind].step(
                                control_outputs[ind, :]
                            )
                        else:
                            observation = env.reset()
                            done = False
                            reward = -1.0
                        this_inputs[ind, :] = observation
                        not_dones[ind] = float(not (done))
                        rewards[ind] += reward
                    this_inputs = (
                        this_inputs.flatten()
                    )  # to match required dimensionality
                    episode_len += 1
                    # print 'this inputs',np.sum(np.isnan(this_inputs))

            # average reward/time step over all episodes
            # rewards /= EVOL_P['max_episode_len']
            rewards /= float(EVOL_P["max_num_episodes"])

            if EVOL_MODE:
                # stepping generation to produce new population
                _, summary = sess.run(
                    [tfs.step_generation, tfs.merged_summary], {fits_pl: rewards}
                )
                writer.add_summary(summary, gen)

                # occasionally displaying stats
                if gen == 0 or (gen + 1) % 1 == 0:
                    print(
                        gen + 1,
                        " ",
                        np.max(rewards),
                        " ",
                        np.mean(rewards),
                        " ",
                        end="\n",
                    )
                    with tf.variable_scope("population", reuse=True):
                        np.save(
                            "./logs/best_at_" + str(gen + 1),
                            sess.run(tf.get_variable("pop"))[0],
                        )

        if EVOL_MODE:
            # writing out graph to visualize
            writer.add_graph(sess.graph)

            with tf.variable_scope("population", reuse=True):
                return sess.run(tf.get_variable("pop"))[0]


if __name__ == "__main__":
    if len(sys.argv) == 1:
        EVOL_MODE = True
        from tensorflow.python.client import device_lib

        local_device_protos = device_lib.list_local_devices()
        print([x.name for x in local_device_protos if x.device_type == "GPU"])

        print("In EVOL_MODE - evolving for", task_name)
        best_net = evol_ctrnn_pong()
        np.save("best_net", best_net)

    else:
        print("Not in EVOL_MODE; Simulating the individual in ", end=" ")
        EVOL_MODE = False
        EVOL_P["pop_size"] = 1
        EVOL_P["max_gens"] = 1
        EVOL_P["evaluation_duration"] = 5000
        EVOL_P["max_num_episodes"] = 1

        best_gen_filename = str(sys.argv[-1])
        print(best_gen_filename)
        best_genotype = np.load(best_gen_filename)
        # print(np.shape(best_genotype))
        evol_ctrnn_pong(best_genotype)

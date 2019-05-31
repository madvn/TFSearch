"""
Evolving CTRNNs for openAI Gym's CartPole-v0
Author - Madhavun Candadai
Date Created - Oct 3, 2017
"""
import gym
import tensorflow as tf
import numpy as np
import sys
from TFSearch import *
from CTRNN import *
from params import *


def evol_ctrnn_cartPole(genotype_to_sim=None):
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
            with tf.variable_scope("population", reuse=True):
                sess.run(
                    tf.assign(
                        tf.get_variable("pop"),
                        tf.convert_to_tensor(np.asarray([genotype_to_sim])),
                    )
                )
            # opening file to write behavioral data
            behavior_file = open("PB_full.dat", "w")

        # stepping through the generations
        for gen in range(EVOL_P["max_gens"]):
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

            # simulate a number of episodes to evaluate fitness
            for episode_ind in range(EVOL_P["max_num_episodes"]):
                # Reset task status - Setting all CTRNNs as not done with task
                not_dones = np.ones(EVOL_P["pop_size"])

                # Reset envs - Initializing inputs for the CTRNNs for each episode
                this_inputs = []
                for env in envs:
                    this_inputs = np.append(this_inputs, env.reset())

                # Reset CTRNNs - random initial conditions for CTRNNs
                sess.run(
                    ctrnns.set_states,
                    {state_pl: np.zeros((1, EVOL_P["pop_size"] * EVOL_P["cns_size"]))},
                )
                sess.run(
                    ctrnns.set_outputs,
                    {
                        output_pl: np.ones((1, EVOL_P["pop_size"] * EVOL_P["cns_size"]))
                        * 0.5
                    },
                )

                episode_len = 0
                while any(not_dones) and episode_len < EVOL_P["max_episode_len"]:
                    if not EVOL_MODE:
                        behavior_file.write(
                            np.append((this_inputs, sess.run(ctrnns.state_v)))
                        )
                    # at least one CTRNN is still balancing the pole or episode_len limit is not reached
                    # Simulate network
                    sess.run(
                        [ctrnns.set_external_inputs],
                        feed_dict={input_pl: [this_inputs]},
                    )

                    sess.run([ctrnns.step_state, ctrnns.step_output])
                    control_outputs = sess.run(ctrnns.control_outputs)
                    control_outputs = control_outputs[:, 1] - control_outputs[:, 0]
                    control_outputs[control_outputs > 0.5] = 1
                    control_outputs[control_outputs < 0.5] = 0

                    # identify valid envs to step
                    valid_env_inds = np.argwhere(not_dones).flatten()

                    # Construct the inputs for next time step by stepping valid envs
                    this_inputs = np.zeros((EVOL_P["pop_size"], EVOL_P["obs_size"]))
                    for valid_env_ind in valid_env_inds:
                        if not EVOL_MODE:
                            env.render()
                        observation, reward, done, _ = envs[valid_env_ind].step(
                            int(control_outputs[valid_env_ind])
                        )
                        this_inputs[valid_env_ind, :] = observation * (1 - done)
                        not_dones[valid_env_ind] = float(not (done))
                        rewards[valid_env_ind] += reward
                    this_inputs = (
                        this_inputs.flatten()
                    )  # to match required dimensionality
                    episode_len += 1
                    if not EVOL_MODE:
                        print(episode_ind, episode_len)
                # print episode_len,' ',
                # print '\n'

            # average reward/time step over all episodes
            rewards /= EVOL_P["max_episode_len"]
            # rewards /= EVOL_P['max_num_episodes']

            if EVOL_MODE:
                # stepping generation to produce new population
                _, summary = sess.run(
                    [tfs.step_generation, tfs.merged_summary], {fits_pl: rewards}
                )
                writer.add_summary(summary, gen)

                # occasionally displaying stats
                if gen == 0 or (gen + 1) % 1 == 0:
                    print(gen + 1, " ", np.max(rewards), " ", np.mean(rewards))

        if EVOL_MODE:
            # writing out graph to visualize
            writer.add_graph(sess.graph)

            with tf.variable_scope("population", reuse=True):
                return sess.run(tf.get_variable("pop"))[0]
        else:
            return rewards


if __name__ == "__main__":
    if len(sys.argv) == 1:
        EVOL_MODE = True
        from tensorflow.python.client import device_lib

        local_device_protos = device_lib.list_local_devices()
        print([x.name for x in local_device_protos if x.device_type == "GPU"])

        print("In EVOL_MODE - evolving for", task_name)
        best_net = evol_ctrnn_cartPole()
        print("Saving best")
        np.save("best_net", best_net)

        # Simulating best
        EVOL_MODE = False
        EVOL_P["pop_size"] = 1
        EVOL_P["max_gens"] = 1
        reward = evol_ctrnn_cartPole(best_net)
        print("Average reward = ", reward)

    else:
        print("Not in EVOL_MODE; Simulating the individual in ")
        EVOL_MODE = False
        EVOL_P["pop_size"] = 1
        EVOL_P["max_gens"] = 1
        EVOL_P["max_num_episodes"] = 400
        best_gen_filename = str(sys.argv[-1])
        print(best_gen_filename)
        best_genotype = np.load(best_gen_filename)
        # print np.shape(best_genotype)
        reward = evol_ctrnn_cartPole(best_genotype)
        print("Average reward = ", reward)

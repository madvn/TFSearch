'''
Evolving CTRNNs for openAI Gym's CartPole-v0
Author - Madhavun Candadai
Date Created - Oct 3, 2017
'''
import gym
import tensorflow as tf
import numpy as np
from TFSearch import *
from CTRNN import *
from params import *
def preprocess_observation(obs):
    obs = obs[35:194] # crop
    obs = obs[::2,::2,0] # downsample
    obs[obs != 0] = 1 # binarize
    obs=obs[:,4:76] # removing side borders - size is now (80,72)
    obs = obs.flatten()
    return obs

def evol_ctrnn_pong():
    # init random population
    this_pop = np.random.uniform(size=(EVOL_P['pop_size'],EVOL_P['genotype_size']),
                                low=EVOL_P['genotype_min_val'],
                                high=EVOL_P['genotype_max_val'])

    # Initializing the environments
    print 'Initializing the environments...'
    envs = []
    for _ in range(EVOL_P['pop_size']):
        envs.append(gym.make('Pong-v0'))

    #print '\n\n\nDefault graph = ',(tf.get_default_graph())
    #print tf.get_default_session()
    writer = tf.summary.FileWriter('./logs/')

    # Initializing poriton of the graph for generation stepping
    tfs = TFSearch(EVOL_P)

    # Initializing poriton of the graph for genotype_to_phenotype conversion for whole population
    make_net = MakeCTRNN(EVOL_P)

    # Initializing poriton of the graph for CTRNN stepping for the whole population
    ctrnns = CTRNN(EVOL_P)

    # Initialize portion of the graph for preprocessing the inputs - using numpy instead
    '''with tf.name_scope('preprocessing'):
        rgb_input = tf.placeholder(tf.float32.shape=(201,160,3))
        gray_input = tf.squeeze(tf.image.rgb_to_grayscale(rgb_input)) # converting to gray and making it 2-D
        _,game_area,_ = tf.slice(gray_input,[34,161,-1]) # cropping to only include game region and not scoreboard
        preprocessed_input = tf.reshape(game_area,[-1])
    '''

    # Running TFSearch
    with tf.Session(config=tf.ConfigProto(log_device_placement=False)) as sess:
        # stepping through the generations
        for gen in range(EVOL_P['max_gens']):
            # Initializing reward vector
            rewards = np.zeros(EVOL_P['pop_size'])

            # make phenotypes
            net_weights,net_biases,net_taus,net_in_weights = sess.run([make_net.weight_matrices,
                                                                        make_net.bias_matrices,
                                                                        make_net.tau_matrices,
                                                                        make_net.in_weight_matrices],{pop_pl:this_pop})

            # set phenotypes before starting simulations
            sess.run([tf.assign(ctrnns.weights_T,net_weights),
                        tf.assign(ctrnns.biases_T,net_biases),
                        tf.assign(ctrnns.taus_T,net_taus),
                        tf.assign(ctrnns.in_weights,net_in_weights)])

            # simulate a number of episodes to evaluate fitness
            for episode_ind in range(EVOL_P['max_num_episodes']):
                # Reset task status - Setting all CTRNNs as not done with task
                not_dones = np.ones(EVOL_P['pop_size'])

                # Reset envs - Initializing inputs for the CTRNNs for each episode
                this_inputs = []
                for env in envs:
                    _ = env.reset()
                    # fire ball to begin game
                    observation = preprocess_observation(env.step(1)[0])
                    this_inputs = np.append(this_inputs,observation)

                # Reset CTRNNs - random initial conditions for CTRNNs
                this_states = [np.random.rand(EVOL_P['cns_size']*EVOL_P['pop_size'])]
                this_outputs = [np.random.rand(EVOL_P['cns_size']*EVOL_P['pop_size'])]

                episode_len = 0
                while any(not_dones) and episode_len<EVOL_P['max_episode_len']:
                    # at least one CTRNN is still balancing the pole or episode_len limit is not reached
                    # Simulate network
                    this_states,this_outputs = sess.run([ctrnns.step_state,ctrnns.step_output],{state_pl:this_states,output_pl:this_outputs,input_pl:[this_inputs]})

                    # contruct output from CTRNNs
                    control_outputs = np.reshape(this_outputs,(EVOL_P['pop_size'],EVOL_P['cns_size']))[:,-1]
                    control_outputs[control_outputs > 0.5] = 1
                    control_outputs[control_outputs < 0.5] = 0

                    # identify valid envs to step
                    valid_env_inds = np.argwhere(not_dones).flatten()

                    # Construct the inputs for next time step by stepping valid envs
                    this_inputs = np.zeros((EVOL_P['pop_size'],EVOL_P['obs_size']))
                    for valid_env_ind in valid_env_inds:
                        observation, reward, done, _ = envs[valid_env_ind].step(int(control_outputs[valid_env_ind]))
                        # Preprocessing observation - might be faster than tensorflow?
                        observation = preprocess_observation(observation)
                        this_inputs[valid_env_ind,:] = observation
                        not_dones[valid_env_ind] = float(not(done))
                        rewards[valid_env_ind] += reward
                    this_inputs = this_inputs.flatten() # to match required dimensionality
                    episode_len += 1
                #print episode_len,' ',
                #print '\n'

            # average reward/time step over all episodes
            rewards /= EVOL_P['max_episode_len']
            #rewards /= EVOL_P['max_num_episodes']

            # stepping generation to produce new population
            this_pop = sess.run(tfs.step_generation,{pop_pl:this_pop,fits_pl:rewards})

            # occasionally displaying stats
            if gen==0 or (gen+1)%1==0:
                print gen+1, ' ', np.max(rewards), ' ', np.mean(rewards)

        # writing out graph to visualize
        writer.add_graph(sess.graph)

    return this_pop[0]

if __name__ == "__main__":
    best_net = evol_ctrnn_pong()
    EVOL_P['pop_size'] = 1

    # simulating the best network
    with tf.Session() as sess:
        with tf.variable_scope('simulating_best'):
            #sess.run(tf.reset_default_graph())
            with MakeCTRNN(EVOL_P) as make_net:
                net_weights,net_biases,net_taus,net_in_weights = sess.run([make_net.weight_matrices,make_net.bias_matrices,make_net.tau_matrices,make_net.in_weight_matrices],{pop_pl:[best_net]})

            env = gym.make('CartPole-v0')

            this_inputs = env.reset()
            this_states = [np.random.rand(EVOL_P['cns_size']*EVOL_P['pop_size'])]
            this_outputs = [np.random.rand(EVOL_P['cns_size']*EVOL_P['pop_size'])]
            print np.shape(this_outputs),'  ',np.shape(net_weights)

            rewards = 0.
            with CTRNN(EVOL_P) as ctrnns:
                # set phenotypes before starting simulations
                sess.run([tf.assign(ctrnns.weights_T,net_weights),
                            tf.assign(ctrnns.biases_T,net_biases),
                            tf.assign(ctrnns.taus_T,net_taus),
                            tf.assign(ctrnns.in_weights,net_in_weights)])

                for episode_ind in range(100):
                    this_inputs = env.reset()
                    this_states = [np.random.rand(EVOL_P['cns_size']*EVOL_P['pop_size'])]
                    this_outputs = [np.random.rand(EVOL_P['cns_size']*EVOL_P['pop_size'])]

                    done = False
                    episode_len = 0
                    while not(done) and episode_len<500:
                        this_states,this_outputs = sess.run([ctrnns.step_state,ctrnns.step_output],{state_pl:this_states,output_pl:this_outputs,input_pl:[this_inputs]})

                        # contruct output from CTRNNs
                        control_outputs = this_outputs[:,-1]
                        control_outputs[control_outputs > 0.5] = 1
                        control_outputs[control_outputs <= 0.5] = 0

                        # Construct the inputs for next time step by stepping valid envs
                        #env.render()
                        this_inputs, reward, done, _ = env.step(int(control_outputs[0]))
                        episode_len += 1
                        rewards += reward
                        #print rewards,done,episode_ind

        print 'Average reward out of 100 episodes from 200 sim_steps for the best net = ',rewards/EVOL_P['max_num_episodes']
        np.savetxt('bestCartPoleBalancer.dat',best_net)

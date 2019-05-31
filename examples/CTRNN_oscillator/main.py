'''
This main function sets up the experiment and creates objects of the TFSearch
and CTRNN_popSim classes to run the evolutionary search.
The goal is to evolve a 2-neuron CTRNN oscillator
Author: Madhavun Candadai
Date Created: Sep 23, 2017
'''
import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from TFSearch import *
from CTRNN import *
from params import *

def evol_CTRNN_osc(fitness_file_name,plot_flag):
    # initialize a population
    this_pop = np.random.uniform(size=(EVOL_P['pop_size'],EVOL_P['genotype_size']),
                    low=EVOL_P['genotype_min_val'],
                    high=EVOL_P['genotype_max_val'])

    # random initial conditions for CTRNN
    this_state = [np.random.rand(EVOL_P['cns_size']*EVOL_P['pop_size'])]
    this_output = [np.random.rand(EVOL_P['cns_size']*EVOL_P['pop_size'])]

    # max(fitness) vs Gens vector
    fits = []

    # create TensorFlow session
    with tf.Session() as sess:
        # run evol search
        with TFSearch(EVOL_P) as tfs:
            for g in range(EVOL_P['max_gens']):
                with MakeCTRNN(EVOL_P) as make_net:
                    # first simply convert to phenotype so the weights can then be set and not be sent eevry time
                    net_weights, net_biases, net_taus = sess.run([make_net.weight_matrices,make_net.bias_matrices,make_net.tau_matrices],feed_dict={pop_pl:this_pop})

                # init net with phenotypes so it doesn't have to be sent with a placeholder every time.
                with CTRNN(EVOL_P,net_weights,net_biases,net_taus) as net:
                    # run networks for a while to avoid transient
                    for t in range(int(EVOL_P['transient_duration'])):
                        this_state, this_output = sess.run([net.step_state,net.step_output],feed_dict={state_pl:this_state,output_pl:this_output})

                    fs = np.zeros(EVOL_P['pop_size'])
                    for t in range(int(EVOL_P['eval_duration'])):
                        this_state, this_output_p = sess.run([net.step_state,net.step_output],feed_dict={state_pl:this_state,output_pl:this_output})

                        # absolute time-difference of neuron 1 in all networks
                        out1_abs_diff = np.reshape(np.fabs(this_output - this_output_p),[EVOL_P['pop_size'],EVOL_P['cns_size']])
                        # summing absolute time difference over time for all networks
                        fs += out1_abs_diff.swapaxes(0,1)[0]

                        # replacing new output with present output
                        this_output = this_output_p

                    # normalizing fitness
                    fs /= EVOL_P['eval_duration']
                    fits.append(np.max(fs))

                    # displaying stats intermittently
                    if g==0 or (g+1)%20==0:
                        print (g+1),'\t',np.max(fs)

                    # create new population based on fitness
                    this_pop = sess.run(tfs.step_generation,feed_dict={pop_pl:this_pop,fits_pl:fs})

                    if g == 0:
                        writer = tf.summary.FileWriter('./logs/')
                        writer.add_graph(sess.graph)
                # saving max(fitness) over gens
                np.savetxt(fitness_file_name,fits)

        # Simulating and plotting best net only if required
        if plot_flag:
            # change params to reflect the new population with only best individual
            EVOL_P_pop_size = EVOL_P['pop_size']
            EVOL_P['pop_size'] = 1

            # Init and simulate best network
            print 'Simulating best network'
            this_pop = [this_pop[0]] # first individual is always the best because TFSearch sorts them

            # random initial conditions for CTRNN
            this_state = [np.random.rand(EVOL_P['cns_size']*EVOL_P['pop_size'])]
            this_output = [np.random.rand(EVOL_P['cns_size']*EVOL_P['pop_size'])]

            with MakeCTRNN(EVOL_P) as make_net:
                # Convert to phenotype
                net_weights, net_biases, net_taus = sess.run([make_net.weight_matrices,make_net.bias_matrices,make_net.tau_matrices],feed_dict={pop_pl:this_pop})

            with CTRNN(EVOL_P,net_weights,net_biases,net_taus) as net:
                outputMat = []
                for t in range(int(EVOL_P['eval_duration'])*5):
                    this_state, this_output = sess.run([net.step_state,net.step_output],feed_dict={state_pl:this_state,output_pl:this_output})
                    outputMat.append(this_output[0])

                '''print 'Plotting best network'
                for i in range(EVOL_P['cns_size']):
                    plt.plot(np.asarray(outputMat)[:,i])
                plt.xlabel('Time')
                plt.ylabel('Neuron outputs')
                plt.savefig('./outputs.png')
                plt.close()
                '''
            # Resetting pop_size for future runs, if any
            EVOL_P['pop_size'] = EVOL_P_pop_size


    # returning max(fitness) vs Gens for this evolSearch run
    return fits


if __name__ == "__main__":
    # set up experiment here...
    # run 'num_evol_runs' times and plot only the first
    for r in range(EVOL_P['num_evol_runs']):
        print '\nRun #',r+1
        fVec = evol_CTRNN_osc('fitness_'+str(r+1)+'.dat',r==0)
        #plt.plot(fVec)
    '''plt.xlabel('Generations')
    plt.ylabel('Fitness')
    plt.savefig('./fitsvsGens.png')
    '''

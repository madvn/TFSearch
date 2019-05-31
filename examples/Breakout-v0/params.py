'''
File with all required parameters.
Note the required parameters for TFSearch - See TFSearch.py
'''
import numpy as np

cns_size = 50+1
step_size = .8
obs_size = 80*72
## Parameters dict - Could possibly split into multiple dicts - env+agent+evol
EVOL_P = {# task-specific params
            'max_num_episodes':100,
            'max_episode_len':200,
            'obs_size':obs_size,
            # CTRNN params
            'cns_size':cns_size,
            'step_size':step_size,
            # required params for TFSearch
            'genotype_size':cns_size**2+2*cns_size+obs_size*cns_size,
            'genotype_min_val':0,
            'genotype_max_val':1,
            'scaling_high':np.hstack([[2]*cns_size,[5]*cns_size,[5]*cns_size**2,[5]*obs_size*cns_size]), #upper bound [taus,biases,weights]
            'scaling_low':np.hstack([[1]*cns_size,[-5]*cns_size,[-5]*cns_size**2,[-5]*obs_size*cns_size]), #lower bound [taus,biases,weights]
            'pop_size':100,
            'max_gens':5,
            'elitist_fraction':0.1,
            'mutation_variance':0.1,
            'crossover_fraction':0.1,
            'crossover_probability': 0.3}

if __name__ == "__main__":
    print 'Printing all params...'
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(EVOL_P)

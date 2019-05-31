# [WIP] TFSearch - A starter package for evolutionary optimization using Tensorflow

This repo has code that has a fully-vectorized implementation of an evolutionary algorithm in Tensorflow that can fully take advantage of GPUs' super-fast matrix transformation abilities. TFSearch.py can be used by anyone with custom genotype-phenotype conversion and fitness evaluation operations (preferably in Tensorflow for best results) to perform evolutionary optimization for any task.

A quick introduction to evolutionary optimization can be found [here](https://github.com/madvn/TFSearch/blob/master/evol_intro.md)

This package has mostly been written with the aim of optimizing [CTRNN](https://github.com/madvn/CTRNN)s using evolutionary search on GPUs.

See example for optimizing a CTRNN to produce oscillations [here](https://github.com/madvn/TFSearch/tree/master/examples/CTRNN_oscillator).

TODO
- [ ] Packagify
- [ ] Make and push tests
- [ ] Test examples, add figures and docs

## Getting Started
Download TFSearch.py and save it some place you can import it from and then in your script
```
from TFSearch import *
```
The dependencies are <link to numpy> and <link to Tensorflow>. TFSearch is implemented such that the init() defines the graph for step_generation. Therefore, the graph can be instantiated by creating an object of the TFSearch class.
```
tfs = TFSearch(EVOL_P)
```
where *EVOL_P* is a dictionary that requires the following
```
EVOL_P = {#required params for TFSearch
            'genotype_size':47,
            'genotype_min_val':0,
            'genotype_max_val':1,
            'pop_size':100,
            'elitist_fraction':0.1,
            'mutation_variance':0.1,
            'crossover_fraction':0.1,
            'crossover_probability': 0.3,
            'device':'/gpu:0'}
```
The object, *tfs*, while defining the graph for step_generation, also initializes the population tf.variable, *pop* under the variable_scope *population*, and provides a tf.placeholder *fits_pl* for the vector of fitnesses. Now, the evolutionary algorithm can be written as
```
with tf.Session() as sess:
    tf.global_variables_initializer().run() #required to initialize the population variable

    # run for set number of generations
    for gen in range(max_gens):
        phenotypes = convert_genotype_to_phenotype() #could read from tf.get_variable('pop') and convert to phenotypes
        # sess.run([convert_genotype_to_phenotype]) #could write Tensorflow code to read from tf.get_variable('pop')

        fitness = evaluate_fitness(phenotypes) #evaluate fitnesses of phenotypes to give vector of fitnesses for the population
        # fitness = sess.run([evaluate_fitness]) #could write Tensorflow code to evaluate phenotypes, possibly for the whole population at once

        sess.run(tfs.step_generation,feed_dict={fits_pl:fitness})

    # get the set of parameters corresponding to the best solution
    with tf.variable_scope('population'):
        best_solution = sess.run(tf.get_variable('pop'))[0] # since the population is always ranked by TFSearch
```
The [examples](https://github.com/madvn/TFSearch/tree/master/examples) folder shows different implementations of *convert_genotype_to_phenotype* and *evaluate_fitness* for different task and how one might use Tensorflow for implementing those functions.

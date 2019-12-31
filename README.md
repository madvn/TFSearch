# TFSearch - A starter package for evolutionary optimization using Tensorflow

This repo has code that has a fully-vectorized implementation of an evolutionary algorithm in Tensorflow that can fully take advantage of GPUs' super-fast matrix transformation abilities. TFSearch.py can be used by anyone with custom genotype-phenotype conversion and fitness evaluation operations (preferably in Tensorflow for best results) to perform evolutionary optimization for any task.

A quick introduction to evolutionary optimization can be found [here](https://github.com/madvn/TFSearch/blob/master/evol_intro.md)

This package has mostly been written with the aim of optimizing [CTRNN](https://github.com/madvn/CTRNN)s using evolutionary search on GPUs.

See example for optimizing a CTRNN to produce oscillations [here](https://github.com/madvn/TFSearch/tree/master/examples/CTRNN_oscillator).

TODO
- [ ] pop_pl can be a tf.Variable
- [ ] provide fitness function to TFSearch
- [ ] Packagify
- [ ] Tests
- [X] Test examples, add figures and docs

## Getting Started
Download TFSearch.py and save it some place you can import it from and then in your script
```
from TFSearch import *
```
The dependencies are numpy and Tensorflow. TFSearch is implemented such that the init() defines the graph for `step_generation`. Therefore, the graph can be instantiated by creating an object of the TFSearch class.
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
            }
```
Importing * from TFSearch and creating the object provides the following to the user

1. pop_pl - a placeholder for the population
2. fits_pl - a placeholder for the fitness values of the population
3. step_generation - a tensorflow graph operation that uses the fitness values and the current population to return a new population of solutions.

The user needs to write their own function (preferably in Tensorflow) to estimate the fitness of the population for the task at hand. Note that the genotype is in the range `[0,1]` in the params shown above - the user might have to scale them before evaluating fitness. Once a fitness function has been defined, a typical search can be conducted as follows -

```
# initialize a random population
this_pop = np.random.uniform(
    size=(EVOL_P["pop_size"], EVOL_P["genotype_size"]),
    low=EVOL_P["genotype_min_val"],
    high=EVOL_P["genotype_max_val"],
)

# create TensorFlow session
sess = tf.Session()

# creating the TFSearch graph
tfs = TFSearch(EVOL_P)

# run for set number of generations
for gen in range(EVOL_P['max_gens']):
    # ******* user-defined function depending on the task
    fitness = evaluate_fitness(this_pop)
    # OR could write Tensorflow code to evaluate phenotypes, possibly for the whole population at once
    # fitness = sess.run([evaluate_fitness], feed_dict={pop_pl:this_pop})

    # now using TFSearch to create a new populaton based on their fitness
    this_pop = sess.run(tfs.step_generation,feed_dict={pop_pl:this_pop, fits_pl:fitness})

# Since the population is always ordered, the best individual is always the first
best_solution = this_pop[0]
```

The [examples](https://github.com/madvn/TFSearch/tree/master/examples) folder shows a few examples of using this package.

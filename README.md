# TFSearch - A starter package for evolutionary optimization using Tensorflow

This repo has code that has a fully-vectorized implementation of an evolutionary algorithm in Tensorflow that can fully take advantage of GPUs' super-fast matrix transformation abilities. TFSearch.py can be used by anyone with custom genotype-phenotype conversion and fitness evaluation operations (preferably in Tensorflow for best results) to perform evolutionary optimization for any task.

## A quick note on evolutionary optimization
Evolutionary optimization is a biased search of the parameter space starting with a random population of possible solutions. The search occurs over several generations during which the population mimics the natural phenomenon of the "survival of the fittest". Since this is a generic optimization procedure and is not specific to neural networks, the term "system" has been used in place of "neural network" in the future. Each parameter set in the population, called the *genotype*, is assigned a fitness score based on how good the system performs given that set of parameters. An instantiation of a system based on a particular genotype is called the *phenotype*. Individuals that perform well are retained over generations and they produce children thereby improving the quality of solutions in the population over generations. The search is terminated after a pre-determined number of generations or after at least one solution with the desired fitness is found.

The basic programming structure for any evolutionary optimization would look something like this:
```
this_pop = create_random_population(population_size,genotype_size) # a random matrix of size populationSize X genotypeSize
for generation in range(max_number_of_generations):
    phenotypes = convert_genotype_to_phenotype(this_pop) # building systems defined by the genotypes
    fitnesses = evaluate_fitness(phenotypes) # returns vector of fitnesses with shape=(1,population_size)
    this_pop = step_generation(this_pop,fitnesses) # new population based on fitness of current population
best_solution = this_pop[0] # TFSearch always ranks the population by fitness
```

This Tensorflow package, *TFSearch*, takes care of *create_random_population()*, and *step_generation*. Most importantly, TFSearch steps through the different steps of creating a new population based on fitness, as a sequence of matrix transformations on the population matrix thereby taking full advantage of Tensorflow and GPU computation. Since *convert_genotype_to_phenotype* and *evaluate_fitness* depend on the problem/task and the components of the genotype, any evolutionary optimization package would require the user to define those functions, and so does this package. The user may also write *convert_genotype_to_phenotype* and *evaluate_fitness* using Tensorflow such that the entire population is processed as a matrix. However, it is not required that Tensorflow is used for these functions. The <link to examples> folder show use cases with different implementations for *convert_genotype_to_phenotype* and *evaluate_fitness*. Here's a quick look at the functions performed by TFSearch.step_generation() in order to produce a new population, given the current population and their fitness values.

### Elitist selection
After fitness evaluation, the population is ranked according to fitness and a fraction of the best performing genotypes, given by *elitist_fraction*, are kept as is for the next generation. These genotypes are also the parents from whom offspring will be produced to generate a new population for the next generation. These parents are first copied over to create a population the same size as the original population. These copies are referred to as the children and these children undergo crossover and mutation.

### Crossover
This operation selects a particular subset of children from the population, given by crossover_fraction and shuffles values between them at each point in the genotype, with a probability given by crossover_probability. This allows the genotypes to arbitrarily shift within the parameter space thereby contributing the exploration component of the search.
Note that *crossover_fraction* is the fraction of children and not a fraction of the population. Children are the number of individuals in the population that were copied from the elite parents and therefore *num_children = pop_size - floor(elitist_fraction*pop_size)*

### Mutation
The process of mutation adds noise to the genotypes such that the genotypes are offset by a distance that is an N-dimensional normal distribution with mean at the genotype and variance given by *mutation_variance*, where N is dimensionality of the genotype. All children are subject to mutation at every generation step while producing a new population.

Upon completion of these above steps, TFSearch concatenates the crossed over and mutated children with the elitist parents and assigns the new population to be evaluated for fitness.

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
The <link to examples> folder shows different implementations of *convert_genotype_to_phenotype* and *evaluate_fitness* for different task and how one might use Tensorflow for implementing those functions.

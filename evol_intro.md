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

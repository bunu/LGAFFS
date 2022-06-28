# Lexicographic Genetic Algorithm for Fair Feature Selection (LGAFFS)

LGAFFS is a lexicographic GA for feature selection that uses both a measure of accruracy (Geometric Mean of Sensitivity and Specificity) and 4 different measures of fairness (Discrimination Score, Consistency, False Positive error rate balance, False Negative Error Rate Ballance) to select features for a base classifier using the wrapper approach.

## GA Pseudocode

```
max_iterations
attribute_number
population_size
min_prob
max_prob
crossover_rate
mutation_rate
mutation_probability
accuracy_epsilon
fairness_rank_epsilon % This could be a statistical test in the future, however KIS
fairness_test_epsilon
classifier
best_individual
3-fold cross-validation

Individual:
    genes (bitstring % attribute selection information)
    classification_result
    accuracy_measure
    fitness_values

    initialiseRamped(p)
	individual = Individual
    	for gene in Individual.genes:
    	    if random < p:
	    gene = 1

GA()
    population = initialise_population()
    create_folds()
    evolve_ga()

initialise_population()
    step_size = (max_prob-min_prob) / pop_size
    for i in population_size:
    	p = min_prob + i*step_size
	population += Individual.initialise(p)
    return population

evolve_ga()
    for i in max_iterations:
    	if stagnation:
            stop
    	for individual in population:
    	    run_classifier_with_cross_validation()
    	calculate_fitness_measures()
    	top_n_pop = population_lexicographic_top_n()
    	best_individual = best_individual or ranked_pop[0]
    	generate_population(ranked_pop)

generate_population(population, top_n_pop)
    new_pop = []
    new_pop += top_n_pop % Top n solutions carried over, elitist selection
    while new_pop < population_size:
        individual1 = tournamentselection()
	individual2 = tournamentselection()
        if random <= crossover_rate:
	   individual1, individual2 = uniform_crossover()
	for individual1.gene, individual2.gene: % Mutation should is per gene not per individual
	   if random <= mutation_rate: 
	      mutation()
	new_pop += individual1,individual2

tournament_selection()
    select 2 individuals at random
    if not |i1_accuracy - i2_accuracy| > accuracy_epsilon:
       calculate_fairness_permutations()
       if i1_fairness - i2_fairness > fairness_win_epsilon:
       	  return fairest_individual
    return best_accuracy_individual       

calculate_fairness_permutations()
    for fair_measure in measures:
        if i1_measure - i2_measure > fairness_test_epsilon:
            add_win_to_individual()
    return permutation_results

population_lexicographic_top_n()
    lexicographic_rank = []
    accuracy_rank = sort_population_by_accuracy()
    while lexicographic_rank length < n:
        individual = accuracy_rank.head()
        individuals = select_all_individuals_within_accuracy_epsilon()
    calculate_average_rank_of_fairness_permutations()
    if (individual.average_rank - fairest_individual.average_rank) > fairness_rank_epsilon
       lexicographic_rank.add(fairest_individual)
       accuracy_rank.remove(fairest_individual)
    else
       lexicographic_rank.add(individual)
       accuracy_rank.remove(individual)
```

## Lexicographic top n example
The following method illistrates how LGAFFS can select the top n individuals. By default LGAFFS has an elitism of 1, where only the best individual is retained.

```
population | accuracy
i1 0.9
i2 0.89
i3 0.88
i4 0.87
i5 0.86
i6 0.835
i7 0.7
...
in 0.1

accuracy_epsilon = 0.05
fairness_rank_epsilon = 0.5
n = 3

ROUND 1
5 individuals within accuracy epsilon so calculate fairness average ranking

i1 = 1.8
i2 = 1.2
i3 = 3
i4 = 4
i5 = 5

i2 (fairest) - i1 (most accurate) is greater than fairness_rank_epsilon,
i2 is promoted

lexicographic_rank = [i2]
lexicographic_rank length < n

ROUND 2

4 individuals within accuracy epsilon
i1 = 1.55
i3 = 1.45
i4 = 3
i5 = 4

i3 - i1 is less than fairness_rank_epsilon, i1 is promoted.
lexicographic_rank = [i2,i1]
lexicographic_rank length < n

ROUND 3

4 individuals within accuracy epsilon
i3 = 2
i4 = 3
i5 = 4
i6 = 1

i6 - i3 is greater than fairness_rank_epsilon, i6 is promoted

lexicographic_rank [i2,i1,i6]
lexicographic_rank length == n
STOP
```

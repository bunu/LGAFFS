import random
from typing import List
from typing import Tuple
from sklearn.model_selection import StratifiedKFold
from pandas import DataFrame

from baseline_individual import Individual
from Classifier import Classifier
from encoder import Encoder


class base_GA:
    attribute_number: int
    population_size: int
    max_iterations: int
    crossover_rate: float
    mutation_rate: float
    classifier: Classifier
    encoder: Encoder
    num_folds: int
    data: DataFrame
    folds = List[Tuple[DataFrame, DataFrame]]
    population: List[Individual]
    best_individual: Individual
    elitism_n: int
    out: str

    def __init__(self, attribute_number: int, classifier: Classifier, data: DataFrame, encoder: Encoder, out: str,
                 population_size=100, max_iterations=50, crossover_rate=0.9, mutation_rate=0.05, num_folds=3,
                 elitism_n=1, seed=1):
        self.attribute_number = attribute_number
        self.classifier = classifier
        self.encoder = encoder
        self.data = data
        self.num_folds = num_folds
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = []
        self.best_individual = Individual(attribute_number)
        self.folds = []
        self.elitism_n = elitism_n
        self.out = out
        random.seed(seed)
        skf = StratifiedKFold(n_splits=self.num_folds, shuffle=True, random_state=1)
        for train, test in skf.split(data.iloc[:, :-1], data.iloc[:, -1]):
            self.folds.append((data.iloc[train, :], data.iloc[test, :]))

    def initialise_individual(self, p: float) -> Individual:
        individual = Individual(self.attribute_number)
        individual.initialise_genes(p)
        return individual

    def initialise_ramped_population(self, min_p: float, max_p: float) -> None:
        p_step = (max_p - min_p) / self.population_size
        for i in range(0, self.population_size):
            individual = self.initialise_individual(min_p + i * p_step)
            self.population.append(individual)

    def tournament_selection(self) -> Individual:
        individual1 = random.choice(self.population)
        individual2 = random.choice(self.population)
        return individual1 if individual1 > individual2 else individual2

    def uniform_crossover(self, individual1: Individual, individual2: Individual) -> Tuple[Individual, Individual]:
        child1 = Individual(self.attribute_number)
        child2 = Individual(self.attribute_number)
        for i in range(0, self.attribute_number):
            if random.random() > 0.5:
                child1.genes[i] = individual1.genes[i]
                child2.genes[i] = individual2.genes[i]
            else:
                child1.genes[i] = individual2.genes[i]
                child2.genes[i] = individual1.genes[i]
        return child1, child2

    def mutation(self, individual: Individual) -> None:
        for i in range(0, self.attribute_number):
            if random.random() < self.mutation_rate:
                individual.genes[i] = 1 - individual.genes[i]

    def generate_population(self, top_n_pop: List[Individual]) -> None:
        new_pop = []
        new_pop.extend(top_n_pop)
        while len(new_pop) <= self.population_size:
            individual1 = self.tournament_selection()
            individual2 = self.tournament_selection()
            if random.random() < self.crossover_rate:
                individual1, individual2 = self.uniform_crossover(individual1, individual2)
            self.mutation(individual1)
            self.mutation(individual2)
            if individual1.validate_genes() and individual2.validate_genes():
                new_pop.extend((individual1, individual2))
        self.population = new_pop

    def evolve_ga(self) -> None:
        raise NotImplementedError

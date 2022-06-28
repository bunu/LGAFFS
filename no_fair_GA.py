from dataclasses import dataclass
from pandas import DataFrame

from base_GA import base_GA
from util import calculate_and_print_statistics, run_classifier_with_cross_validation
from util import log_individual
from baseline_individual import Individual
from Classifier import Classifier
from encoder import Encoder


@dataclass
class IndividualRank:
    individual: Individual
    rank: float


class no_fair_GA(base_GA):

    def __init__(self, attribute_number: int, classifier: Classifier, data: DataFrame, encoder: Encoder, out: str,
                 population_size=100, max_iterations=50, crossover_rate=0.9, mutation_rate=0.05, num_folds=3,
                 elitism_n=1, seed=1):
        super().__init__(attribute_number, classifier, data, encoder, out, population_size, max_iterations,
                         crossover_rate, mutation_rate, num_folds, elitism_n, seed)

    def evolve_ga(self) -> None:
        for i in range(0, self.max_iterations):
            index = 0
            for individual in self.population:
                log_individual(individual, "Iteration %s, Individual %s" % (i, index))
                index += 1
                run_classifier_with_cross_validation(individual, self.folds, self.encoder, self.classifier)
            calculate_and_print_statistics(self.population, "Iteration %s" % i)
            self.population.sort(key=lambda x: x.accuracy, reverse=True)
            self.best_individual = self.best_individual if self.best_individual > self.population[0]\
                else self.population[0]
            self.generate_population(self.population[0:self.elitism_n])
            log_individual(self.best_individual, "Best Individual", printIndividual=True)

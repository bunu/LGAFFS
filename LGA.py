import random
from dataclasses import dataclass
from typing import List
from typing import Tuple
from pandas import DataFrame
from CONSTANTS import ACCURACY_EPSILON
from CONSTANTS import FAIRNESS_RANK_EPSILON
from CONSTANTS import FAIRNESS_COMPARISON_EPSILON
from base_GA import base_GA
from util import calculate_and_print_statistics, run_classifier_with_cross_validation
from util import log_individual
from LGAIndividual import LGAIndividual
from Classifier import Classifier
from encoder import Encoder


@dataclass
class IndividualRank:
    individual: LGAIndividual
    rank: float


class LGA(base_GA):
    population: List[LGAIndividual]

    def __init__(self, attribute_number: int, classifier: Classifier, data: DataFrame, encoder: Encoder, out: str,
                 population_size=100, max_iterations=50, crossover_rate=0.9, mutation_rate=0.05, num_folds=3,
                 elitism_n=1, seed=1):
        super().__init__(attribute_number, classifier, data, encoder, out, population_size, max_iterations,
                         crossover_rate, mutation_rate, num_folds, elitism_n, seed)

    def initialise_individual(self, p: float) -> LGAIndividual:
        individual = LGAIndividual(self.attribute_number)
        individual.initialise_genes(p)
        return individual

    @staticmethod
    def _get_individuals_within_epsilon(population: List[LGAIndividual]) -> List[IndividualRank]:
        individuals = [IndividualRank(population[0], 0)]
        stop = False
        i = 1
        while not stop and i < len(population):
            if population[0].accuracy - population[i].accuracy < ACCURACY_EPSILON:
                individuals.append(IndividualRank(population[i], 0))
                i += 1
            else:
                stop = True
        return individuals

    @staticmethod
    def _compare_individuals(i1: List[float], i2: List[float]) -> float:
        for v1, v2 in zip(i1, i2):
            if abs(v1 - v2) > FAIRNESS_COMPARISON_EPSILON:
                return v1 - v2
        return 0

    @staticmethod
    def _calculate_rank(permutation: List[str], individual_rank_list: List[IndividualRank]) -> None:
        permValues = []
        for iRank in individual_rank_list:
            permValues.append((iRank, [iRank.individual.fairness_values[p] for p in permutation]))
        for i in range(0, len(permValues)-1):
            for j in range(i+1, len(permValues)):
                if LGA._compare_individuals(permValues[i][1], permValues[j][1]) < 0:
                    temp = permValues[i]
                    permValues[i] = permValues[j]
                    permValues[j] = temp
        rank = 1
        for val in permValues:
            val[0].rank += rank
            rank += 1

    def _calculate_average_rank_of_fairness_permutations(self, rank_list: List[IndividualRank]) -> None:
        if len(rank_list) == 1:
            return
        permutations = LGAIndividual.calculate_permutations(list(rank_list[0].individual.fairness_values.keys()))
        for p in permutations:
            self._calculate_rank(p, rank_list)
        for i in rank_list:
            i.rank = i.rank / len(permutations)

    def population_lexicographic_top_n(self) -> List[LGAIndividual]:
        lexicographic_rank = []
        accuracy_rank = self.population.copy()
        accuracy_rank.sort(key=lambda x: x.accuracy, reverse=True)
        while len(lexicographic_rank) < self.elitism_n:
            top_individual = accuracy_rank[0]
            individual_rank_list = self._get_individuals_within_epsilon(accuracy_rank)
            self._calculate_average_rank_of_fairness_permutations(individual_rank_list)
            top_individual_rank = individual_rank_list[0].rank
            individual_rank_list.sort(key=lambda x: x.rank)
            if top_individual_rank - individual_rank_list[0].rank > FAIRNESS_RANK_EPSILON:
                lexicographic_rank.append(individual_rank_list[0].individual)
                accuracy_rank.remove(individual_rank_list[0].individual)
            else:
                lexicographic_rank.append(top_individual)
                accuracy_rank.remove(top_individual)
        return lexicographic_rank

    def tournament_selection(self) -> LGAIndividual:
        individual1 = random.choice(self.population)
        individual2 = random.choice(self.population)
        return individual1 if individual1 > individual2 else individual2

    def uniform_crossover(self, individual1: LGAIndividual, individual2: LGAIndividual) \
            -> Tuple[LGAIndividual, LGAIndividual]:
        child1 = LGAIndividual(self.attribute_number)
        child2 = LGAIndividual(self.attribute_number)
        for i in range(0, self.attribute_number):
            if random.random() > 0.5:
                child1.genes[i] = individual1.genes[i]
                child2.genes[i] = individual2.genes[i]
            else:
                child1.genes[i] = individual2.genes[i]
                child2.genes[i] = individual1.genes[i]
        return child1, child2

    def mutation(self, individual: LGAIndividual) -> None:
        for i in range(0, self.attribute_number):
            if random.random() < self.mutation_rate:
                individual.genes[i] = 1 - individual.genes[i]

    def generate_population(self, top_n_pop: List[LGAIndividual]) -> None:
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
        for i in range(0, self.max_iterations):
            index = 0
            for individual in self.population:
                log_individual(individual, "Iteration %s, Individual %s" % (i, index))
                index += 1
                run_classifier_with_cross_validation(individual, self.folds, self.encoder, self.classifier)
            calculate_and_print_statistics(self.population, "Iteration %s" % i)
            top_n_pop = self.population_lexicographic_top_n()
            self.best_individual = self.best_individual if self.best_individual > top_n_pop[0] else top_n_pop[0]
            self.generate_population(top_n_pop)
            log_individual(self.best_individual, "Best Individual", printIndividual=True)

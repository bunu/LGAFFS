import baseline_individual

from CONSTANTS import ACCURACY_WEIGHT
from CONSTANTS import FAIRNESS_WEIGHT


class WeightedIndividual(baseline_individual.Individual):

    def __init__(self, attribute_number: int):
        super().__init__(attribute_number)

    def get_performance(self):
        print("get_performance called")
        fair_sum = 0
        for metric, value in self.fairness_values.items():
            fair_sum += value
        return self.accuracy * ACCURACY_WEIGHT + fair_sum/len(self.fairness_values) * FAIRNESS_WEIGHT

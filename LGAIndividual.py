import baseline_individual
from typing import List
from CONSTANTS import FAIRNESS_COMPARISON_EPSILON
from CONSTANTS import ACCURACY_EPSILON
from CONSTANTS import FAIRNESS_TEST_EPSILON


class LGAIndividual(baseline_individual.Individual):

    @staticmethod
    def calculate_permutations(lst: List[str]) -> List[List[str]]:
        if len(lst) == 1:
            return [lst]
        perm_lst = []
        for i in range(0, len(lst)):
            m = lst[i]
            remainder_lst = lst[:i] + lst[i + 1:]
            for p in LGAIndividual.calculate_permutations(remainder_lst):
                perm_lst.append([m] + p)
        return perm_lst

    def __init__(self, attribute_number: int):
        super().__init__(attribute_number)

    def _fair_comparison(self, other) -> int:
        permutations = LGAIndividual.calculate_permutations(list(self.fairness_values.keys()))
        win_self = 0
        win_other = 0
        for p in permutations:
            for metric in p:
                if self.fairness_values.get(metric) - other.fairness_values.get(metric) > FAIRNESS_COMPARISON_EPSILON:
                    win_self += 1
                    break
                if other.fairness_values.get(metric) - self.fairness_values.get(metric) > FAIRNESS_COMPARISON_EPSILON:
                    win_other += 1
                    break
        if abs(win_self - win_other) < FAIRNESS_TEST_EPSILON:
            return 0
        else:
            return win_self - win_other

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LGAIndividual):
            return NotImplemented
        return (abs(self.accuracy - other.accuracy) < ACCURACY_EPSILON) and self._fair_comparison(other) == 0

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, LGAIndividual):
            return NotImplemented
        return __eq__(self, other) or self.accuracy - other.accuracy >= ACCURACY_EPSILON \
            or self._fair_comparison(other) > 0

    def __le__(self, other: object) -> bool:
        if not isinstance(other, LGAIndividual):
            return NotImplemented
        return __eq__(self, other) or other.accuracy - self.accuracy > ACCURACY_EPSILON \
            or self._fair_comparison(other) < 0

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, LGAIndividual):
            return NotImplemented
        return self.accuracy - other.accuracy > ACCURACY_EPSILON or self._fair_comparison(other) > 0

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, LGAIndividual):
            return NotImplemented
        return other.accuracy - self.accuracy > ACCURACY_EPSILON or self._fair_comparison(other) > 0

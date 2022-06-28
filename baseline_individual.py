from random import random
from typing import List
from typing import Dict


class Individual:

    accuracy: float
    roc_auc: float
    rank:  float
    classification_results: List[object]
    fairness_values: Dict[str, float]
    confusion_matrix: Dict[str, float]

    def __init__(self, attribute_number: int):
        self.genes = [0] * attribute_number
        self.accuracy = 0
        self.roc_auc = 0
        self.rank = 0
        self.classification_results = []
        self.fairness_values = {}
        self.confusion_matrix = {}

    def validate_genes(self) -> bool:
        for x in self.genes:
            if x:
                return True
        return False

    def initialise_genes(self, p: float) -> None:
        for i in range(0, len(self.genes)):
            if random() < p:
                self.genes[i] = 1
        if not self.validate_genes():
            self.initialise_genes(p)

    def get_performance(self):
        return self.accuracy

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return abs(self.get_performance() - other.get_performance()) == 0

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.get_performance() - other.get_performance() >= 0

    def __le__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.get_performance() - other.get_performance() <= 0

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.get_performance() - other.get_performance() > 0

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, Individual):
            return NotImplemented
        return self.get_performance() - other.get_performance() < 0

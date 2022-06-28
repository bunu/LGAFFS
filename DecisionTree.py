import numpy as np
from sklearn.tree import DecisionTreeClassifier
from pandas import DataFrame
from Classifier import Classifier


class DecisionTree(Classifier):

    classifier: DecisionTreeClassifier

    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = DecisionTreeClassifier(**kwargs)

    def _fit(self, Xs: DataFrame, Y: DataFrame):
        self.classifier.fit(Xs, Y)

    def _predict(self, Xs: DataFrame) -> np.ndarray:
        return self.classifier.predict(Xs)

    def _predict_proba(self, Xs: DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(Xs)

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from pandas import DataFrame
from Classifier import Classifier


class RandomForest(Classifier):

    classifier: RandomForestClassifier

    def __init__(self, **kwargs):
        super().__init__()
        self.classifier = RandomForestClassifier(**kwargs)

    def _fit(self, Xs: DataFrame, Y: DataFrame):
        self.classifier.fit(Xs, Y)

    def _predict(self, Xs: DataFrame) -> np.ndarray:
        return self.classifier.predict(Xs)

    def _predict_proba(self, Xs: DataFrame) -> np.ndarray:
        return self.classifier.predict_proba(Xs)

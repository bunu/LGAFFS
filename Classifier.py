from math import sqrt
from typing import List

import numpy as np
from pandas import DataFrame
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder

from util import generate_confusion_matrix


class Classifier:

    labelEncoder: LabelEncoder
    classifier: BaseEstimator

    def __init__(self, **kwargs):
        self.labelEncoder = LabelEncoder()

    def _fit(self, Xs: DataFrame, Y: DataFrame):
        pass

    def _predict(self, Xs: DataFrame) -> np.ndarray:
        pass

    def _predict_proba(self, Xs: DataFrame) -> np.ndarray:
        pass

    def build_classifier(self, Xs: DataFrame, Y: DataFrame, encoder: ColumnTransformer = None) -> None:
        if encoder:
            Xs = encoder.transform(Xs)
        self.labelEncoder.fit(Y.values.ravel())
        Y = self.labelEncoder.transform(Y.values.ravel())
        self._fit(Xs, Y)

    def predict(self, Xs: DataFrame, encoder: ColumnTransformer = None) -> List[object]:
        if encoder:
            Xs = encoder.transform(Xs)
        return self.labelEncoder.inverse_transform(self._predict(Xs))

    def accuracy(self, Xs: DataFrame, Y: DataFrame, class_attribute: str, positive_class: str,
                 encoder: ColumnTransformer = None) -> float:
        return self.geometric_sensitivity_specificity(Xs, Y, class_attribute, positive_class, encoder)

    def geometric_sensitivity_specificity(self, Xs: DataFrame, Y: DataFrame, class_attribute: str, positive_class: str,
                                          encoder: ColumnTransformer = None) -> float:
        if encoder:
            Xs = encoder.transform(Xs)
        predictions = self._predict(Xs)
        predictions = self.labelEncoder.inverse_transform(predictions)
        Y['Prediction'] = predictions
        tp, fp, tn, fn = generate_confusion_matrix(Y, class_attribute, "Prediction", positive_class)
        sensitivity = len(tp) / (len(tp) + len(fn))
        specificity = len(tn) / (len(tn) + len(fp))
        return sqrt(sensitivity * specificity)

    def roc_auc(self, Xs: DataFrame, Y: DataFrame, encoder: ColumnTransformer = None) -> float:
        if encoder:
            Xs = encoder.transform(Xs)
        Y = self.labelEncoder.transform(Y.values.ravel())
        return roc_auc_score(Y, self._predict_proba(Xs)[:, 1])

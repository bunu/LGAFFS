import logging
from dataclasses import dataclass
from typing import List, Tuple
from numpy import inf
from numpy import max
from numpy import min
from pandas import DataFrame
from pandas import concat

from Classifier import Classifier
from baseline_individual import Individual
from encoder import Encoder
from fairness_metrics import calculate_fairness_measures


@dataclass
class stat:
    min: int
    max: int
    sum: int


def calculate_and_print_statistics(population: List[Individual], tag: str) -> None:
    stats = {
        "accuracy": stat(inf, 0, 0),
        "discrimination_score": stat(inf, 0, 0),
        "consistency": stat(inf, 0, 0),
        "FPERBS": stat(inf, 0, 0),
        "FNERBS": stat(inf, 0, 0)
    }
    for individual in population:
        stats.get("accuracy").sum += individual.accuracy
        stats.get("accuracy").min = min([stats.get("accuracy").min, individual.accuracy])
        stats.get("accuracy").max = max([stats.get("accuracy").max, individual.accuracy])
        for key in individual.fairness_values.keys():
            stats.get(key).sum += individual.fairness_values.get(key)
            stats.get(key).min = min([stats.get(key).min, individual.fairness_values.get(key)])
            stats.get(key).max = max([stats.get(key).max, individual.fairness_values.get(key)])

    print("Population Statistics: %s:" % tag)
    for key in stats.keys():
        print("%s Min: %s, Average: %s, Max: %s" % (key, stats.get(key).min, stats.get(key).sum / len(population),
                                                    stats.get(key).max))


def log_individual(ind: Individual, tag: str, printIndividual: bool = False) -> None:
    s = "%s: Genes: %s, Accuracy: %s, ROC_AUC: %s, Fairness: %s, ConfusionMatrix: %s" \
        % (tag, ind.genes, ind.accuracy, ind.roc_auc, ind.fairness_values, ind.confusion_matrix)
    if printIndividual:
        print(s)
    logging.info(s)


def generate_confusion_matrix(data: DataFrame, class_attribute: str, predicted_value: str, positive_class: str) \
        -> Tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    p = data[data[class_attribute] == positive_class]
    tp = p[p[class_attribute] == p[predicted_value]]
    fn = p[p[class_attribute] != p[predicted_value]]
    n = data[data[class_attribute] != positive_class]
    tn = n[n[class_attribute] == n[predicted_value]]
    fp = n[n[class_attribute] != n[predicted_value]]
    return tp, fp, tn, fn


def run_classifier_with_cross_validation(individual: Individual, folds: List[Tuple[DataFrame, DataFrame]],
                                         encoder: Encoder, classifier: Classifier) -> None:
    series_index = []
    accuracy_sum = 0
    for i in range(0, len(individual.genes)):
        if individual.genes[i] == 1:
            series_index.append(i)
    for train, test in folds:
        trainModified = train.take(series_index, axis="columns")
        testModified = test.take(series_index, axis="columns")
        localEncoder = encoder.custom_transformer(list(trainModified.columns))
        localEncoder.fit(concat([trainModified, testModified]))
        classifier.build_classifier(trainModified, train.take([-1], axis="columns"), encoder=localEncoder)
        individual.classification_results.append(classifier.predict(testModified, encoder=localEncoder))
        accuracy_sum += classifier.accuracy(testModified, test.take([-1], axis="columns"), encoder.class_attribute,
                                            encoder.positive_class, encoder=localEncoder)
        individual.accuracy = accuracy_sum / len(folds)
        calculate_fairness_measures(individual, folds, encoder)

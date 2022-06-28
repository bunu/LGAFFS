#!/usr/bin/env python3

import logging
import argparse
import sys
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedKFold

import attributeParser
from CONSTANTS import FOLDS
from Classifier import Classifier
from baseline_individual import Individual
from DecisionTree import DecisionTree
from RandomForest import RandomForest
from LGA import LGA
from encoder import Encoder
from fairness_metrics import fairness_measures
from no_fair_GA import no_fair_GA
from util import log_individual
from weighted_fair_GA import weighted_fair_GA
from sequentialForwardSelection import SequentialForwardSelection


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("target_attribute")
    parser.add_argument("target_val")
    parser.add_argument("sensitive_attribute")
    parser.add_argument("sensitive_val")
    parser.add_argument("-c", "--cross_validation", action='store_true')
    parser.add_argument("--algorithm", default="DecisionTree")
    parser.add_argument("--GA", default="lex_GA")
    parser.add_argument("-t", "--test_input")
    parser.add_argument("--log_level", default="WARNING")
    parser.add_argument("--random_state", default="1")
    parser.add_argument("--attribute_file")
    parser.add_argument("--na_args")
    parser.add_argument("--pop_min_p", default="0.1")
    parser.add_argument("--pop_max_p", default="0.5")
    args = parser.parse_args()

    numeric_level = getattr(logging, args.log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.log_level)
    logging.basicConfig(filename="fairFeatureSelection.log", level=numeric_level)

    args.target_val = args.target_val.strip("'")
    try:
        args.target_val = int(args.target_val)
    except ValueError:
        logging.info("parseArgs: target_val is not a number")

    try:
        args.sensitive_val = int(args.sensitive_val)
    except ValueError:
        logging.info("parseArgs: sensitive_val is not a number")
    return args


def run_algorithm(algorithm: Classifier, data: DataFrame, encoder: Encoder, args: argparse.Namespace) -> Individual:
    ga_switcher = {
        "no_fair_GA": no_fair_GA,
        "lex_GA": LGA,
        "weighted_fair_GA": weighted_fair_GA,
    }
    galgo = ga_switcher.get(args.GA)
    _, features = data.shape
    ga = galgo(features - 1, algorithm, data, encoder, "out")
    ga.initialise_ramped_population(float(args.pop_min_p), float(args.pop_max_p))
    ga.evolve_ga()
    return ga.best_individual


def calculate_fairness_measures(individual: Individual, testData: DataFrame, encoder: Encoder) -> None:
    data = testData.assign(Prediction=individual.classification_results)
    fairness_measures(individual, data, encoder)


def test_model(individual: Individual, algorithm: Classifier, trainData: DataFrame, testData: DataFrame,
               encoder: Encoder) -> None:
    series_index = []
    for i in range(0, len(individual.genes)):
        if individual.genes[i] == 1:
            series_index.append(i)
    trainModified = trainData.take(series_index, axis="columns")
    testModified = testData.take(series_index, axis="columns")
    localEncoder = encoder.custom_transformer(list(trainModified.columns))
    localEncoder.fit(pd.concat([trainModified, testModified]))
    algorithm.build_classifier(trainModified, trainData.take([-1], axis="columns"), encoder=localEncoder)
    individual.classification_results = algorithm.predict(testModified, encoder=localEncoder)
    individual.accuracy = algorithm.accuracy(testModified, testData.take([-1], axis="columns"), encoder.class_attribute,
                                             encoder.positive_class, encoder=localEncoder)
    individual.roc_auc = algorithm.roc_auc(testModified, testData.take([-1], axis="columns"), encoder=localEncoder)
    calculate_fairness_measures(individual, testData, encoder)


def sequential_forward_selection(data: DataFrame, algorithm: Classifier, encoder: Encoder, args: argparse.Namespace):
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=int(args.random_state))
    i = 1
    for train, test in skf.split(data.iloc[:, :-1], data.iloc[:, -1]):
        _, features = data.shape
        train = data.iloc[train, :]
        test = data.iloc[test, :]
        encoder.fit(data)
        train_Xs = encoder.transform(train)[:, : -1]
        test_Xs = encoder.transform(test)[:, : -1]

        # Training
        sfs = SequentialForwardSelection(algorithm.classifier)
        sfs.fit(train_Xs, train.take([-1], axis=1))
        individual = Individual(features - 1)
        individual.genes = [int(x) for x in sfs.get_support(True).tolist()]

        # Testing
        trainModified = train_Xs[:, individual.genes]
        testModified = test_Xs[:, individual.genes]
        algorithm.build_classifier(trainModified, train.take([-1], axis="columns"))
        individual.classification_results = algorithm.predict(testModified)
        individual.accuracy = algorithm.accuracy(testModified, test.take([-1], axis="columns"),
                                                 encoder.class_attribute, encoder.positive_class)
        individual.roc_auc = algorithm.roc_auc(testModified, test.take([-1], axis="columns"))
        calculate_fairness_measures(individual, test, encoder)

        # Logging
        log_individual(individual, "Cross Validation %s test partition result: " % i, True)
        i += 1


def baseline_algorithm(data: DataFrame, algorithm: Classifier, encoder: Encoder, args: argparse.Namespace):
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=int(args.random_state))
    i = 1
    for train, test in skf.split(data.iloc[:, :-1], data.iloc[:, -1]):
        _, features = data.shape
        train = data.iloc[train, :]
        test = data.iloc[test, :]
        encoder.fit(data)
        train_Xs = encoder.transform(train)[:, : -1]
        test_Xs = encoder.transform(test)[:, : -1]
        individual = Individual(features - 1)
        algorithm.build_classifier(train_Xs, train.take([-1], axis="columns"))
        individual.classification_results = algorithm.predict(test_Xs)
        individual.accuracy = algorithm.accuracy(test_Xs, test.take([-1], axis="columns"), encoder.class_attribute,
                                                 encoder.positive_class)
        individual.roc_auc = algorithm.roc_auc(test_Xs, test.take([-1], axis="columns"))
        calculate_fairness_measures(individual, test, encoder)
        log_individual(individual, "Cross Validation %s test partition result: " % i, True)
        i += 1


def main():
    args = parse_args()
    data = pd.read_csv(args.input, sep=",", na_values=args.na_args)
    encoder = Encoder(attributeParser.parse_attribute_file(args.attribute_file), args.target_attribute, args.target_val,
                      args.sensitive_attribute, args.sensitive_val)
    algorithm_switcher = {
        "DecisionTree": DecisionTree,
        "RandomForest": RandomForest,
    }
    algorithm = algorithm_switcher.get(args.algorithm)
    if args.GA == "sfs":
        sequential_forward_selection(data, algorithm(), encoder, args)
    elif args.GA == "none":
        baseline_algorithm(data, algorithm(), encoder, args)
    elif args.cross_validation:
        skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=int(args.random_state))
        i = 1
        for train, test in skf.split(data.iloc[:, :-1], data.iloc[:, -1]):
            individual = run_algorithm(algorithm(), data.iloc[train, :], encoder, args)
            test_model(individual, algorithm(), data.iloc[train, :], data.iloc[test, :], encoder)
            log_individual(individual, "Cross Validation %s test partition result: " % i, True)
            i += 1
    else:
        try:
            test = pd.read_csv(args.test_input, sep=",")
        except ValueError:
            logging.error("main: No test set given and -cv not set")
            sys.exit(1)
        individual = run_algorithm(algorithm(), data, encoder, args)
        test_model(individual, algorithm(), data, test, encoder)
        log_individual(individual, "Test partition result: ", True)


if __name__ == "__main__":
    main()

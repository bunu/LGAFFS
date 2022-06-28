from sklearn.neighbors import NearestNeighbors
from sklearn import preprocessing
from pandas import DataFrame
from pandas import concat
from typing import Tuple, List

from baseline_individual import Individual
from attributeParser import binarySplitAttribute
from attributeParser import thresholdAttribute
from attributeParser import attributeInfo
from encoder import Encoder
from util import generate_confusion_matrix


def _binary_split_converter(x: str, binary_split_info: binarySplitAttribute) -> str:
    if x in binary_split_info.protected:
        return binary_split_info.protectedName
    else:
        return binary_split_info.unprotectedName


def _threshold_converter(x: float, threshold_info: thresholdAttribute) -> str:
    if x > threshold_info.threshold:
        return threshold_info.highName
    else:
        return threshold_info.lowName


def _convert_protected_attribute(data: DataFrame, attribute_info: attributeInfo, sensitive_attribute: str) -> DataFrame:
    thresholds = [attribute for attribute in attribute_info.threshold if attribute.name == sensitive_attribute]
    if thresholds:
        data = data.assign(temp=[_threshold_converter(x, thresholds[0]) for x in data[sensitive_attribute]])
        data = data.drop(sensitive_attribute, axis=1)
        data = data.rename(columns={"temp": sensitive_attribute})
        return data
    binary_splits = [attribute for attribute in attribute_info.binarySplit if attribute.name == sensitive_attribute]
    if binary_splits:
        data = data.assign(temp=[_binary_split_converter(x, binary_splits[0]) for x in data[sensitive_attribute]])
        data = data.drop(sensitive_attribute, axis=1)
        data = data.rename(columns={"temp": sensitive_attribute})
        return data
    return data


def _false_positive_error_rate(data: DataFrame, class_attribute: str, predicted_value: str, positive_class: str) \
        -> float:
    tp, fp, tn, fn = generate_confusion_matrix(data, class_attribute, predicted_value, positive_class)
    try:
        return len(fp.index) / (len(fp.index) + len(tn.index))
    except ZeroDivisionError:
        return 0


def _false_negative_error_rate(data: DataFrame, class_attribute: str, predicted_value: str, positive_class: str) \
        -> float:
    tp, fp, tn, fn = generate_confusion_matrix(data, class_attribute, predicted_value, positive_class)
    try:
        return len(fn.index) / (len(tp.index) + len(fn.index))
    except ZeroDivisionError:
        return 0


def _calculate_discrimination_values(data: DataFrame, encoder: Encoder, predicted_value: str) \
        -> Tuple[float, float, float, float]:
    data = _convert_protected_attribute(data, encoder.attributes, encoder.sensitive_attribute)
    pdata = data[data[encoder.sensitive_attribute] == encoder.sensitive_value]
    udata = data[data[encoder.sensitive_attribute] != encoder.sensitive_value]
    plen, _ = pdata.shape
    ulen, _ = udata.shape
    pplen, _ = pdata[pdata[predicted_value] == encoder.positive_class].shape
    pulen, _ = udata[udata[predicted_value] == encoder.positive_class].shape
    return pulen, ulen, pplen, plen


def discrimination_score(data: DataFrame, encoder: Encoder, predicted_value: str) -> float:
    pulen, ulen, pplen, plen = _calculate_discrimination_values(data, encoder, predicted_value)
    return (pulen / ulen) - (pplen / plen)


def false_positive_error_rate_balance_score(data: DataFrame, encoder: Encoder,
                                            predicted_value: str) -> float:
    data = _convert_protected_attribute(data, encoder.attributes, encoder.sensitive_attribute)
    pdata = data[data[encoder.sensitive_attribute] == encoder.sensitive_value]
    udata = data[data[encoder.sensitive_attribute] != encoder.sensitive_value]
    return _false_positive_error_rate(udata, encoder.class_attribute, predicted_value, encoder.positive_class) - \
        _false_positive_error_rate(pdata, encoder.class_attribute, predicted_value, encoder.positive_class)


def false_negative_error_rate_balance_score(data: DataFrame, encoder: Encoder, predicted_value: str) -> float:
    data = _convert_protected_attribute(data, encoder.attributes, encoder.sensitive_attribute)
    pdata = data[data[encoder.sensitive_attribute] == encoder.sensitive_value]
    udata = data[data[encoder.sensitive_attribute] != encoder.sensitive_value]
    return _false_negative_error_rate(udata, encoder.class_attribute, predicted_value, encoder.positive_class) - \
        _false_negative_error_rate(pdata, encoder.class_attribute, predicted_value, encoder.positive_class)


def impact_ratio(data: DataFrame, encoder: Encoder, predicted_value) -> float:
    pulen, ulen, pplen, plen = _calculate_discrimination_values(data, encoder, predicted_value)
    try:
        return (pulen / ulen) / (pplen / plen)
    except ZeroDivisionError:
        return -1


def consistency(data: DataFrame, encoder: Encoder, predicted_value: str, k: int):
    x = data.drop([predicted_value, encoder.class_attribute], axis=1)
    y = data[predicted_value]
    encoder.fit(x)
    x = encoder.transform(x)
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(x)
    indices = nbrs.kneighbors(x, return_distance=False)
    return 1 - abs(y - y[indices].mean(axis=1)).mean()


def calculate_fairness_measures(individual: Individual, folds: List[Tuple[DataFrame, DataFrame]],
                                encoder: Encoder) -> None:
    processed_folds = []
    for (_, test), predictions in zip(folds, individual.classification_results):
        test = test.assign(Prediction=predictions)
        processed_folds.append(test)
    data = concat(processed_folds)
    fairness_measures(individual, data, encoder)


def fairness_measures(individual: Individual, data: DataFrame, encoder: Encoder):
    # So to make comparisons easy, we are modifying the values (of all measures other than consistency to only consider
    # the abs value as we consider positive and negative discrimination to be equally as bad and taking this from 1.
    # Therefore all measures are between 0 and 1 with 1 being fair and 0 unfair
    fairness_metrics = {"consistency": consistency(data, encoder, "Prediction", 5),
                        "discrimination_score": 1 - abs(discrimination_score(data, encoder, "Prediction")),
                        "FPERBS": 1 - abs(false_positive_error_rate_balance_score(data, encoder, "Prediction")),
                        "FNERBS": 1 - abs(false_negative_error_rate_balance_score(data, encoder, "Prediction"))
                        }
    individual.fairness_values = fairness_metrics

    # We attach the confusion matrix at this point so that we can record it in our logs later
    tp, fp, tn, fn = generate_confusion_matrix(data, encoder.class_attribute, "Prediction", encoder.positive_class)
    individual.confusion_matrix.update({"TP": len(tp)})
    individual.confusion_matrix.update({"TN": len(tn)})
    individual.confusion_matrix.update({"FP": len(fp)})
    individual.confusion_matrix.update({"FN": len(fn)})

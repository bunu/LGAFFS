import logging
from typing import List
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from pandas import DataFrame

from attributeParser import attributeInfo


class Encoder:

    attributes: attributeInfo
    transformer: ColumnTransformer
    transformations: List[Tuple[str, any, List[str]]]
    class_attribute: str
    positive_class: str
    sensitive_attribute: str
    sensitive_value: str

    def __init__(self, attributes: attributeInfo, class_attribute: str, positive_class: str, sensitive_attribute: str,
                 sensitive_value: str):
        self.attributes = attributes
        self.class_attribute = class_attribute
        self.positive_class = positive_class
        self.sensitive_attribute = sensitive_attribute
        self.sensitive_value = sensitive_value
        self.transformations = []
        for unordered in attributes.unordered:
            self.transformations.append(("cat_%s" % unordered.name, OneHotEncoder(sparse=False), [unordered.name]))
        for ordinal in attributes.ordinal:
            self.transformations.append(("ordinal_%s" % ordinal.name, OrdinalEncoder(categories=[ordinal.values]),
                                         [ordinal.name]))
        self.transformer = ColumnTransformer(transformers=self.transformations, remainder="passthrough")
        logging.info(self.attributes)
        logging.info(self.transformer)

    def fit(self, x: DataFrame):
        self.transformer.fit(x)
        logging.info("transformer.fit()")

    def transform(self, x: DataFrame):
        logging.info("transformer.transform()")
        return self.transformer.transform(x)

    def custom_transformer(self, columnNames: List[str]) -> ColumnTransformer:
        t = []
        for transform in self.transformations:
            if transform[2][0] in columnNames:
                t.append(transform)
        return ColumnTransformer(transformers=t, remainder="passthrough")

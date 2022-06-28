import logging
import sys
from dataclasses import dataclass
from dataclasses import field
from typing import List


@dataclass
class unorderedAttribute:
    name: str

    def __str__(self):
        return self.name


@dataclass
class ordinalAttribute:
    name: str
    values: List[str]

    def __str__(self):
        return self.name


@dataclass
class thresholdAttribute:
    name: str
    lowName: str
    highName: str
    threshold: float

    def __str__(self):
        return self.name


@dataclass
class binarySplitAttribute:
    name: str
    protectedName: str
    unprotectedName: str
    protected: List[str]

    def __str__(self):
        return self.name


@dataclass
class attributeInfo:
    unordered: List[unorderedAttribute] = field(default_factory=list)
    ordinal: List[ordinalAttribute] = field(default_factory=list)
    threshold: List[thresholdAttribute] = field(default_factory=list)
    binarySplit: List[binarySplitAttribute] = field(default_factory=list)


# noinspection PyBroadException
def parse_unordered(attributes: attributeInfo, split_line: List[str]):
    try:
        attributes.unordered.append(unorderedAttribute(split_line[1]))
    except:
        logging.error("Failed to parse unordered attribute %s" % split_line)
        sys.exit(1)


# noinspection PyBroadException
def parse_ordinal(attributes: attributeInfo, split_line: List[str]):
    try:
        attributes.ordinal.append(ordinalAttribute(split_line[1], split_line[2:]))
    except:
        logging.error("Failed to parse ordinal attribute %s" % split_line)
        sys.exit(1)


# noinspection PyBroadException
def parse_threshold(attributes: attributeInfo, split_line: List[str]):
    try:
        attributes.threshold.append(
            thresholdAttribute(split_line[1], split_line[2], split_line[3], float(split_line[4])))
    except:
        logging.error("Failed to parse threshold attribute %s" % split_line)
        sys.exit(1)


# noinspection PyBroadException
def parse_binary_split(attributes: attributeInfo, split_line: List[str]):
    try:
        attributes.binarySplit.append(binarySplitAttribute(split_line[1], split_line[2], split_line[3], split_line[4:]))
    except:
        logging.error("Failed to parse binary split attribute %s" % split_line)
        sys.exit(1)


def default(_attributes: attributeInfo, split_line: List[str]):
    logging.info("Unexpected line (%s), skipping" % split_line)


def parse_attribute_file(filename: str):
    attributes = attributeInfo()

    try:
        f = open(filename, "r")
        lines = f.readlines()
        f.close()
    except IOError:
        logging.info("Unable to read attribute file: %s, it may not exist" % filename)
        return attributes

    switcher = {
        "@unordered": parse_unordered,
        "@ordinal": parse_ordinal,
        "@threshold": parse_threshold,
        "@binarysplit": parse_binary_split
    }

    for line in lines:
        split_line = line.strip().split(" ")
        parse_function = switcher.get(split_line[0], default)
        parse_function(attributes, split_line)
    return attributes

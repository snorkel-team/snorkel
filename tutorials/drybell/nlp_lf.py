"""
This example script replicates functionality from
    Snorkel Drybell (https://arxiv.org/abs/1812.00417)

Similar to Section 5.1, an NLP-based labeling function is
authored using two preprocessors. The LF checks whether the
title or body of an article contains a person mention. The
LF is then executed using Spark, returning a label matrix.
"""


import logging

from pyspark import SparkContext
from pyspark.sql import Row

from snorkel.labeling.apply.lf_applier_spark import SparkLFApplier
from snorkel.labeling.lf import labeling_function
from snorkel.labeling.preprocess import preprocessor
from snorkel.labeling.preprocess.nlp import SpacyPreprocessor
from snorkel.types import DataPoint, FieldMap

logging.basicConfig(level=logging.INFO)


ABSTAIN = 0
NEGATIVE = 1
POSITIVE = 2

DATA = [
    (
        "The sports team won!",
        "In a big sports game, the sports team won. The score is unknown.",
    ),
    (
        "Jane Doe is great.",
        "This article describes how great Jane Doe is. She is great.",
    ),
]


@preprocessor()
def combine_text_preprocessor(title: str, body: str) -> FieldMap:
    return dict(article=f"{title} {body}")


spacy_preprocessor = SpacyPreprocessor("article", "article")


@labeling_function(preprocessors=[combine_text_preprocessor, spacy_preprocessor])
def article_mentions_person(x: DataPoint) -> int:
    for ent in x.article.ents:
        if ent.label_ == "PERSON":
            return ABSTAIN
    return NEGATIVE


def build_lf_matrix() -> None:
    sc = SparkContext()
    rdd = sc.parallelize(DATA)
    rdd = rdd.map(lambda x: Row(title=x[0], body=x[1]))

    applier = SparkLFApplier([article_mentions_person])
    L = applier.apply(rdd)

    logging.info(str(L))


if __name__ == "__main__":
    build_lf_matrix()

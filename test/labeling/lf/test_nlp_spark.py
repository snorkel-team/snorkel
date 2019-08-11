import unittest
from types import SimpleNamespace

import pytest
from pyspark.sql import Row

from snorkel.labeling.lf.nlp import NLPLabelingFunction
from snorkel.labeling.lf.nlp_spark import (
    SparkNLPLabelingFunction,
    spark_nlp_labeling_function,
)
from snorkel.types import DataPoint


def has_person_mention(x: DataPoint) -> int:
    person_ents = [ent for ent in x.doc.ents if ent.label_ == "PERSON"]
    return 0 if len(person_ents) > 0 else -1


@pytest.mark.spark
class TestNLPLabelingFunction(unittest.TestCase):
    def _run_lf(self, lf: SparkNLPLabelingFunction) -> None:
        x = Row(num=8, text="The movie is really great!")
        self.assertEqual(lf(x), -1)
        x = Row(num=8, text="Jane Doe acted well.")
        self.assertEqual(lf(x), 0)

    def test_nlp_labeling_function(self) -> None:
        lf = SparkNLPLabelingFunction(name="my_lf", f=has_person_mention)
        self._run_lf(lf)

    def test_nlp_labeling_function_decorator(self) -> None:
        @spark_nlp_labeling_function()
        def has_person_mention(x: DataPoint) -> int:
            person_ents = [ent for ent in x.doc.ents if ent.label_ == "PERSON"]
            return 0 if len(person_ents) > 0 else -1

        self.assertIsInstance(has_person_mention, SparkNLPLabelingFunction)
        self.assertEqual(has_person_mention.name, "has_person_mention")
        self._run_lf(has_person_mention)

    def test_spark_nlp_labeling_function_with_nlp_labeling_function(self) -> None:
        # Do they have separate _nlp_configs?
        lf = NLPLabelingFunction(name="my_lf", f=has_person_mention)
        lf_spark = SparkNLPLabelingFunction(name="my_lf_spark", f=has_person_mention)
        self.assertEqual(lf(SimpleNamespace(num=8, text="Jane Doe acted well.")), 0)
        self._run_lf(lf_spark)

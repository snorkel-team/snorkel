import unittest
from types import SimpleNamespace

import dill
import pytest

from snorkel.labeling.lf.nlp import NLPLabelingFunction, nlp_labeling_function
from snorkel.preprocess import preprocessor
from snorkel.types import DataPoint


@preprocessor()
def combine_text(x: DataPoint) -> DataPoint:
    x.text = f"{x.title} {x.article}"
    return x


def has_person_mention(x: DataPoint) -> int:
    person_ents = [ent for ent in x.doc.ents if ent.label_ == "PERSON"]
    return 0 if len(person_ents) > 0 else -1


class TestNLPLabelingFunction(unittest.TestCase):
    def _run_lf(self, lf: NLPLabelingFunction) -> None:
        x = SimpleNamespace(
            num=8, title="Great film!", article="The movie is really great!"
        )
        self.assertEqual(lf(x), -1)
        x = SimpleNamespace(num=8, title="Nice movie!", article="Jane Doe acted well.")
        self.assertEqual(lf(x), 0)

    def test_nlp_labeling_function(self) -> None:
        lf = NLPLabelingFunction(name="my_lf", f=has_person_mention, pre=[combine_text])
        self._run_lf(lf)

    def test_nlp_labeling_function_memoized(self) -> None:
        lf = NLPLabelingFunction(name="my_lf", f=has_person_mention, pre=[combine_text])
        lf._nlp_config.nlp.reset_cache()
        self.assertEqual(len(lf._nlp_config.nlp._cache), 0)
        self._run_lf(lf)
        self.assertEqual(len(lf._nlp_config.nlp._cache), 2)
        self._run_lf(lf)
        self.assertEqual(len(lf._nlp_config.nlp._cache), 2)

    @pytest.mark.complex
    def test_labeling_function_serialize(self) -> None:
        lf = NLPLabelingFunction(name="my_lf", f=has_person_mention, pre=[combine_text])
        lf_load = dill.loads(dill.dumps(lf))
        self._run_lf(lf_load)

    def test_nlp_labeling_function_decorator(self) -> None:
        @nlp_labeling_function(pre=[combine_text])
        def has_person_mention(x: DataPoint) -> int:
            person_ents = [ent for ent in x.doc.ents if ent.label_ == "PERSON"]
            return 0 if len(person_ents) > 0 else -1

        self.assertIsInstance(has_person_mention, NLPLabelingFunction)
        self.assertEqual(has_person_mention.name, "has_person_mention")
        self._run_lf(has_person_mention)

    def test_nlp_labeling_function_decorator_no_parens(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing parentheses"):

            @nlp_labeling_function
            def has_person_mention(x: DataPoint) -> int:
                person_ents = [ent for ent in x.doc.ents if ent.label_ == "PERSON"]
                return 0 if len(person_ents) > 0 else -1

    def test_nlp_labeling_function_shared_cache(self) -> None:
        lf = NLPLabelingFunction(name="my_lf", f=has_person_mention, pre=[combine_text])

        @nlp_labeling_function(pre=[combine_text])
        def lf2(x: DataPoint) -> int:
            return 0 if len(x.doc) < 9 else -1

        lf._nlp_config.nlp.reset_cache()
        self.assertEqual(len(lf._nlp_config.nlp._cache), 0)
        self.assertEqual(len(lf2._nlp_config.nlp._cache), 0)
        self._run_lf(lf)
        self.assertEqual(len(lf._nlp_config.nlp._cache), 2)
        self.assertEqual(len(lf2._nlp_config.nlp._cache), 2)
        self._run_lf(lf2)
        self.assertEqual(len(lf._nlp_config.nlp._cache), 2)
        self.assertEqual(len(lf2._nlp_config.nlp._cache), 2)

    def test_nlp_labeling_function_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "different parameters"):

            @nlp_labeling_function()
            def has_person_mention(x: DataPoint) -> int:
                person_ents = [ent for ent in x.doc.ents if ent.label_ == "PERSON"]
                return 0 if len(person_ents) > 0 else -1

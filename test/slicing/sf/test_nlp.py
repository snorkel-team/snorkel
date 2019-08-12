import unittest
from types import SimpleNamespace

from snorkel.preprocess import preprocessor
from snorkel.slicing.sf.nlp import NLPSlicingFunction, nlp_slicing_function
from snorkel.types import DataPoint


@preprocessor()
def combine_text(x: DataPoint) -> DataPoint:
    x.text = f"{x.title} {x.article}"
    return x


def has_person_mention(x: DataPoint) -> int:
    person_ents = [ent for ent in x.doc.ents if ent.label_ == "PERSON"]
    return 0 if len(person_ents) > 0 else -1


class TestNLPSlicingFunction(unittest.TestCase):
    def _run_sf(self, sf: NLPSlicingFunction) -> None:
        x = SimpleNamespace(
            num=8, title="Great film!", article="The movie is really great!"
        )
        self.assertEqual(sf(x), -1)
        x = SimpleNamespace(num=8, title="Nice movie!", article="Jane Doe acted well.")
        self.assertEqual(sf(x), 0)

    def test_nlp_slicing_function(self) -> None:
        sf = NLPSlicingFunction(name="my_sf", f=has_person_mention, pre=[combine_text])
        self._run_sf(sf)

    def test_nlp_slicing_function_decorator(self) -> None:
        @nlp_slicing_function(pre=[combine_text])
        def has_person_mention(x: DataPoint) -> int:
            person_ents = [ent for ent in x.doc.ents if ent.label_ == "PERSON"]
            return 0 if len(person_ents) > 0 else -1

        self.assertIsInstance(has_person_mention, NLPSlicingFunction)
        self.assertEqual(has_person_mention.name, "has_person_mention")
        self._run_sf(has_person_mention)

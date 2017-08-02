import os
import unittest

from snorkel import SnorkelSession
from snorkel.models import Document, candidate_subclass
from snorkel.parser import TSVDocPreprocessor, CorpusParser
from snorkel.parser.spacy_parser import Spacy
from snorkel.candidates import Ngrams, CandidateExtractor
from snorkel.matchers import PersonMatcher
from snorkel.contrib.babble import SemanticParser
import unittest_explanations

class TestBabble(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Failed attempt to make a local snorkel.db
        # os.environ['SNORKELDB'] = 'sqlite://' + os.environ['SNORKELHOME'] + '/test/babble/snorkel.db'
        # print(os.environ['SNORKELDB'])

        # Create initial snorkel.db
        session = SnorkelSession()
        Spouse = candidate_subclass('Spouse', ['person1', 'person2'])

        test_article_path = os.environ['SNORKELHOME'] + '/test/babble/test_article.tsv'
        doc_preprocessor = TSVDocPreprocessor(test_article_path)
        corpus_parser = CorpusParser(parser=Spacy())
        corpus_parser.apply(doc_preprocessor)
        ngrams         = Ngrams(n_max=2)
        person_matcher = PersonMatcher(longest_match_only=True)
        cand_extractor = CandidateExtractor(Spouse, [ngrams, ngrams], [person_matcher, person_matcher], symmetric_relations=True)
        docs = session.query(Document).order_by(Document.name).all()
        sents = [s for doc in docs for s in doc.sentences]
        cand_extractor.apply(sents, split=0)

        cls.candidate_hash = {hash(c): c for c in session.query(Spouse).all()}
        
        # Test candidate: 
        # "City land records show that GM President [Daniel Ammann] and his wife, 
        # [Pernilla Ammann], bought the 15-bedroom mansion on Balmoral Drive in 
        # the upscale historic neighborhood on July 31."

        cls.sp = SemanticParser(Spouse, unittest_explanations.get_user_lists(), 
                                beam_width=10, top_k=-1)

    @classmethod
    def tearDownClass(cls):
        pass

    def check_explanations(self, explanations):
        self.assertTrue(len(explanations))
        for e in explanations:
            if e.candidate and not isinstance(e.candidate, tuple):
                e.candidate = self.candidate_hash[e.candidate]
            LF_dict = self.sp.parse_and_evaluate(e, show_nothing=True)
            if e.semantics:
                self.assertTrue(len(LF_dict['correct']) > 0)
            else:
                self.assertTrue(len(LF_dict['passing']) > 0)
            self.assertTrue(len(LF_dict['correct']) + len(LF_dict['passing']) <= 3)


    def test_logic(self):
        self.check_explanations(unittest_explanations.logic)

    def test_grouping(self):
        self.check_explanations(unittest_explanations.grouping)

    def test_integers(self):
        self.check_explanations(unittest_explanations.integers)

    def test_strings(self):
        self.check_explanations(unittest_explanations.strings)

    def test_lists(self):
        self.check_explanations(unittest_explanations.lists)

    def test_candidate_helpers(self):
        self.check_explanations(unittest_explanations.candidate_helpers)

    def test_index_comparisons(self):
        self.check_explanations(unittest_explanations.index_comparisons)

    def test_pos_ner(self):
        self.check_explanations(unittest_explanations.pos_ner)

    def test_count(self):
        self.check_explanations(unittest_explanations.count)

    def test_absorption(self):
        self.check_explanations(unittest_explanations.absorption)

    def test_anaphora(self):
        self.check_explanations(unittest_explanations.anaphora)

    def test_inversion(self):
        self.check_explanations(unittest_explanations.inversion)

    def test_tuples(self):
        self.check_explanations(unittest_explanations.tuples)


if __name__ == '__main__':
    unittest.main()

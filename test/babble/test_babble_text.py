import os
import unittest

from snorkel import SnorkelSession
from snorkel.models import Document, candidate_subclass
from snorkel.parser import TSVDocPreprocessor, CorpusParser
from snorkel.parser.spacy_parser import Spacy
from snorkel.candidates import Ngrams, CandidateExtractor
from snorkel.matchers import PersonMatcher
from snorkel.contrib.babble import SemanticParser

from test_babble_base import TestBabbleBase
import text_explanations

class TestBabbleText(TestBabbleBase):

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

        cls.candidate_map = {c.get_stable_id(): c for c in session.query(Spouse).all()}
        
        # Test candidate: 
        # "City land records show that GM President [Daniel Ammann] and his wife, 
        # [Pernilla Ammann], bought the 15-bedroom mansion on Balmoral Drive in 
        # the upscale historic neighborhood on July 31."

        cls.sp = SemanticParser(mode='text',
                                candidate_class=Spouse, 
                                user_lists=text_explanations.get_user_lists(),
                                string_format='implicit')

    def test_strings(self):
        self.check_explanations(text_explanations.strings)

    def test_string_lists(self):
        self.check_explanations(text_explanations.string_lists)

    def test_candidate_helpers(self):
        self.check_explanations(text_explanations.candidate_helpers)

    def test_index_words(self):
        self.check_explanations(text_explanations.index_words)
    
    def test_index_chars(self):
        self.check_explanations(text_explanations.index_chars)

    def test_pos_ner(self):
        self.check_explanations(text_explanations.pos_ner)

    def test_count(self):
        self.check_explanations(text_explanations.count)

    def test_anaphora(self):
        self.check_explanations(text_explanations.anaphora)

    def test_inversion(self):
        self.check_explanations(text_explanations.inversion)

    def test_tuples(self):
        self.check_explanations(text_explanations.tuples)

    def test_implicit_strings(self):
        self.check_explanations(text_explanations.implicit_strings)

    def test_spouse_aliases(self):
        self.check_explanations(text_explanations.spouse_aliases)

suite = unittest.TestLoader().loadTestsFromTestCase(TestBabbleText)
unittest.TextTestRunner(verbosity=2).run(suite)
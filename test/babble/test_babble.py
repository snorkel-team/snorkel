import os
import unittest

from snorkel import SnorkelSession
from snorkel.models import Document, candidate_subclass
from snorkel.parser import TSVDocPreprocessor, CorpusParser
from snorkel.parser.spacy_parser import Spacy
from snorkel.candidates import Ngrams, CandidateExtractor
from snorkel.matchers import PersonMatcher
from snorkel.contrib.babble import SemanticParser, Example

class TestBabble(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Failed attempt to make a local snorkel.db
        # os.environ['SNORKELDB'] = 'sqlite://' + os.environ['SNORKELHOME'] + '/test/babble/snorkel.db'
        # print(os.environ['SNORKELDB'])

        # Create initial snorkel.db
        session = SnorkelSession()
        test_article_path = os.environ['SNORKELHOME'] + '/test/babble/test_article.tsv'
        doc_preprocessor = TSVDocPreprocessor(test_article_path)
        corpus_parser = CorpusParser(parser=Spacy())
        corpus_parser.apply(doc_preprocessor)
        Spouse = candidate_subclass('Spouse', ['person1', 'person2'])
        ngrams         = Ngrams(n_max=2)
        person_matcher = PersonMatcher(longest_match_only=True)
        cand_extractor = CandidateExtractor(Spouse, [ngrams, ngrams], [person_matcher, person_matcher])
        docs = session.query(Document).order_by(Document.name).all()
        sents = [s for doc in docs for s in doc.sentences]
        cand_extractor.apply(sents, split=0)
        cls.candidate = None
        for candidate in session.query(Spouse).all():
            if candidate[0].get_span().startswith('Daniel'):
                cls.candidate = candidate
                break
        
        # Test candidate: 
        # "City land records show that GM President [Daniel Ammann] and his wife, 
        # [Pernilla Ammann], bought the 15-bedroom mansion on Balmoral Drive in 
        # the upscale historic neighborhood on July 31."

        user_lists = {
            'colors':['red','green','blue'],
            'bluebird':['blue','bird','fly'],
            'greek':['alpha','beta','gamma'],
            'letters':['a','B','C'],
            'smalls':['a','b','c','d'],
            'spouse':['wife','husband','spouse']}

        cls.sp = SemanticParser(Spouse, user_lists, beam_width=10, top_k=-1)

        

    @classmethod
    def tearDownClass(cls):
        pass

    def test_trivial(self):
        self.assertTrue(True)

    def check_examples(self, examples):
        for e in examples:
            LFs = self.sp.parse_and_evaluate(e)
            self.assertTrue(len(LFs['correct']) + len(LFs['passing']) > 0)

    def test_logic(self):
        examples = [
            # Base
            Example(
                explanation="label True because True",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # And
            Example(
                explanation="label True because True and True",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # Or
            Example(
                explanation="label True because False or True",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # Not boolean
            Example(
                explanation="label True because not False",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # Not function
            Example(
                explanation="label True because 'blue' is not in all caps",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # All
            Example(
                explanation='label True because all of the colors are lowercase',
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # Any
            Example(
                explanation='label True because any of the letters are lowercase',
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # None
            Example(
                explanation='label True because none of the smalls are capitalized',
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),                
        ]
        self.check_examples(examples)

    def test_grouping(self):
        examples = [
            # Parentheses
            Example(
                explanation="label True because True or (True and False)",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
        ]
        self.check_examples(examples)

    def test_integers(self):
        examples = [
            # Equals (Int)
            Example(
                explanation="label True because 1 is equal to 1",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # Integers (digit or text)
            Example(
                explanation="label True because 1 is equal to one",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # Less than
            Example(
                explanation="label True because 1 is less than 2",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # At most
            Example(
                explanation="label True because 2 is less than or equal to 2",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # Greater than
            Example(
                explanation="label True because 2 > 1",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # At least
            Example(
                explanation="label True because 2 is at least 2",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            ]
        self.check_examples(examples)

    def test_strings(self):
        examples = [
            # Equals (String)
            Example(
                explanation="label True because 'yes' equals 'yes'",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # Lowercase
            Example(
                explanation="label True because arg 1 is lowercase",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # Uppercase
            Example(
                explanation="label True because arg 1 is upper case",
                candidate=('FOO', 'bar'),
                denotation=1,
                semantics=None),
            # Capitalized
            Example(
                explanation="label True because arg 1 is capitalized",
                candidate=('Foo', 'bar'),
                denotation=1,
                semantics=None),
            # Starts with
            Example(
                explanation="label True because the word 'blueberry' starts with 'blue'",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # Ends with
            Example(
                explanation="label True because the word 'blueberry' ends with 'berry'",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            ]
        self.check_examples(examples)

    def test_lists(self):
        examples = [
            # In
            Example(
                explanation="label True because 'bar' is in 'foobarbaz'",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # Contains
            Example(
                explanation="label True because the word 'foobarbaz' contains 'oobarba'",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # List
            Example(
                explanation="label True because 'bar' equals 'foo', 'bar', or 'baz'",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # UserList
            Example(
                explanation="label True because 'blue' in colors",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # OrList left
            Example(
                explanation="label True because 'blue' or 'shmoo' is in colors",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # OrList right
            Example(
                explanation="label True because 'blue' ends with 'moe' or 'lue'",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # AndList left
            Example(
                explanation="label True because 'blue' and 'red' are in colors",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # AndList right
            Example(
                explanation="label True because 'blue' contains 'l' and 'u'",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            ]
        self.check_examples(examples)

    def test_candidate_helpers(self):
        examples = [
            # Candidate as string
            Example(
                explanation="label True because argument 1 is 'foo'",
                candidate=('foo', 'bar'),
                denotation=1,
                semantics=None),
            # Left words (list)
            Example(
                explanation="label True because 'wife' is in the words left of arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None), 
            # Right words (list)
            Example(
                explanation="label True because 'wife' is in the words to the right of arg 1",
                candidate=self.candidate,
                denotation=1,
                semantics=None), 
            # Between words (list)
            Example(
                explanation="label True because 'wife' is in the words between arg 1 and arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None), 
            # Sentence (list)
            Example(
                explanation='label True because "wife" is in the sentence',
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            ]
        self.check_examples(examples)

    def test_index_comparisons(self):
        examples = [
            # Index left
            Example(
                explanation="label True because arg 1 is left of arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Index right
            Example(
                explanation="label True because arg 2 is right of arg 1",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Between
            Example(
                explanation="label True because 'wife' is between arg 1 and arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Index left equality
            Example(
                explanation="label True because 'wife' is two words to the left of arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Index left inequality 0
            Example(
                explanation="label True because arg 1 is more than three words to the left of arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Index left inequality 1
            Example(
                explanation="label True because not arg 1 is more than fifty words to the left of arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Index left inequality 2
            Example(
                explanation="label True because ',' is immediately to the left of arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Index left inequality 3
            Example(
                explanation="label True because ',' is right before arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),                      
            # Index within (<=)
            Example(
                explanation="label True because 'wife' is within three words to the left of arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Index OrList left
            Example(
                explanation="label True because 'husband' or 'wife' is within three words to the left of arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Index AndList left
            Example(
                explanation="label True because not 'husband' and 'wife' are within three words to the left of arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Characters0
            Example(
                explanation="label True because 'wife' is less than 10 characters to the left of arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Characters1
            Example(
                explanation="label True because 'wife' is more than 5 characters to the right of arg 1",
                candidate=self.candidate,
                denotation=1,
                semantics=None),            
            ]
        self.check_examples(examples)

    def test_pos_ner(self):
        examples = [
            # Tokens
            Example(
                explanation="label True because at least one word to the left of arg 1 is lower case",
                candidate=self.candidate,
                denotation=1,
                semantics=None), 
            # POS
            Example(
                explanation="label True because at least one noun exists between arg 1 and arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # NER
            Example(
                explanation="label True because there are no people between arg 1 and arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),     
            ]
        self.check_examples(examples)

    def test_count(self):
        examples = [
            # Count0
            Example(
                explanation="label True because there are not three people in the sentence",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Count1
            Example(
                explanation="label True because the number of words between arg 1 and arg 2 is less than 25",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Count2
            Example(
                explanation="label True because there are more than 3 words between arg 1 and arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Count3
            Example(
                explanation="label True because at least one word exists between arg 1 and arg 2",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Count4
            Example(
                explanation="label True because there are two nouns to the left of arg 1",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Count5
            Example(
                explanation="label True because there are less than three nouns to the left of arg 1",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Count6
            Example(
                explanation="label True because there are not more than two nouns to the left of arg 1",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # Count7
            Example(
                explanation="label True because at least one word to the left of arg 2 starts with a spouse word",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            ]
        self.check_examples(examples)

    def test_absorption(self):
        examples = [
                # Partially unparseable
                Example(
                    explanation="label True because 1 is less than 2 and the moon is full",
                    candidate=('foo', 'bar'),
                    denotation=1,
                    semantics=None),
            ]
        self.check_examples(examples)

    def test_anaphora(self):
        examples = [
            # Them
            Example(
                explanation="label True because 'wife' is between arg 1 and arg 2 and 'divorced' is not between them",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            # TODO: he/she, his/her, him/her
            ]
        self.check_examples(examples)

    def test_inversion(self):
        examples = [
            # Inverted sentence
            Example(
                explanation="label True because to the left of arg 2 is a spouse word",
                candidate=self.candidate,
                denotation=1,
                semantics=None),
            ]
        self.check_examples(examples)

    


if __name__ == '__main__':
    unittest.main()

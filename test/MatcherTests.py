import os, requests, sys, unittest, cPickle
from time import sleep
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from snorkel.matchers import *
from snorkel.parser import SentenceParser
from snorkel.candidates import Ngrams

DATA_PATH = os.environ['SNORKELHOME'] + '/test/data/'

class TestMatchers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(DATA_PATH + 'CDR_TestSet_sents.pkl', 'rb') as f:
            cls.CDR_sents = cPickle.load(f)
        cls.sp     = SentenceParser()
        cls.ngrams = Ngrams()

    @classmethod
    def tearDownClass(cls):
        sleep(1)  # TODO: Fix sentence parser so that this hack not necessary!
        cls.sp._kill_pserver()
    
    def test_dictionary_match(self):
        # TODO
        pass

    def test_union(self):
        # TODO
        pass

    def test_concat(self):
        # TODO
        pass

    def test_regex_match(self):
        # TODO
        pass

    def test_slot_fill_match(self):
        
        # Test 1
        dm        = DictionaryMatch(d=['X','Y','Z'])
        rm        = RegexMatchSpan(rgx=r'\d{3}')
        sf        = SlotFillMatch(dm, rm, pattern="{0}-{1}")
        test_sent = "X-123 causes gas."
        sent      = list(self.sp.parse(test_sent))[0]
        matches   = list(sf.apply(self.ngrams.apply(sent)))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].get_span(), "X-123")

        # Test 2
        dm        = DictionaryMatch(d=['Burritos', 'Tacos'], ignore_case=True)
        sf        = SlotFillMatch(dm, pattern="{0} and/or {0}")
        test_sent = "Burritos and/or tacos causes gas."
        sent      = list(self.sp.parse(test_sent))[0]
        matches   = list(sf.apply(self.ngrams.apply(sent)))
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].get_span(), "Burritos and/or tacos")


if __name__ == '__main__':
    unittest.main()

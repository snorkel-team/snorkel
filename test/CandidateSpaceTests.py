import os, sys, unittest, cPickle
from time import sleep
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from snorkel.candidates import *
from snorkel.parser import SentenceParser

DATA_PATH = os.environ['SNORKELHOME'] + '/test/data/'

class TestCandidateSpace(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        with open(DATA_PATH + 'CDR_TestSet_sents.pkl', 'rb') as f:
            cls.sents = cPickle.load(f)
        with open(DATA_PATH + 'CDR_TestSet_sent_0_ngrams.pkl', 'rb') as f:
            cls.gold_ngrams_0 = cPickle.load(f)
        cls.sp     = SentenceParser()

    @classmethod
    def tearDownClass(cls):
        sleep(1)  # TODO: Fix sentence parser so that this hack not necessary!
        cls.sp._kill_pserver()
    
    def test_ngrams(self):
        ngrams = Ngrams(n_max=3, split_tokens=None)
        self.assertEqual(self.gold_ngrams_0, list(ngrams.apply(self.sents[0])))

    def test_split_tokens(self):
        ngrams = Ngrams(n_max=3)
        test_sent = "We found disease A/B in cow Alpha-3."
        sent      = list(self.sp.parse(test_sent))[0]
        ngs       = list(ngrams.apply(sent))
        self.assertEqual(len(ngs), 25)


if __name__ == '__main__':
    unittest.main()

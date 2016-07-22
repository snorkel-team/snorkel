import os, sys, unittest, cPickle
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from snorkel.candidates import *

DATA_PATH = os.environ['SNORKELHOME'] + '/test/data/'

class TestCandidateSpace(unittest.TestCase):
    
    def setUp(self):
        with open(DATA_PATH + 'CDR_TestSet_sents.pkl', 'rb') as f:
            self.sents = cPickle.load(f)

        with open(DATA_PATH + 'CDR_TestSet_sent_0_ngrams.pkl', 'rb') as f:
            self.gold_ngrams_0 = cPickle.load(f)

    def tearDown(self):
        pass

    def test_ngrams(self):
        ngrams = Ngrams(n_max=3)
        self.assertEqual(self.gold_ngrams_0, list(ngrams.apply(self.sents[0])))

if __name__ == '__main__':
    unittest.main()

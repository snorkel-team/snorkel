import os, sys, unittest, cPickle
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ddlite_matchers import *

DATA_PATH = os.environ['DDLHOME'] + '/test/data/'

class TestMatchers(unittest.TestCase):
    
    def setUp(self):
        with open(DATA_PATH + 'CDR_TestSet_sents.pkl', 'rb') as f:
            self.sents = cPickle.load(f)

    def tearDown(self):
        pass
    
    def test_dictionary_matcher(self):
        pass

if __name__ == '__main__':
    unittest.main()

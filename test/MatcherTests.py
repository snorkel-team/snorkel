import os, sys, unittest

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import ddlite_matcher
from ddlite_parser import SentenceParser

class TestMatchers(unittest.TestCase):
    
    def setUp(self):
        
        with open('data_matcher/cc_kurt.txt', 'rb') as f:
            self.txt = f.read()
        sp = SentenceParser()
        self.sents = list(sp.parse(self.txt))
        
        with open('data_matcher/dict1.txt', 'rb') as g:
            self.d1 = [line.strip() for line in g.readlines()]
        
        with open('data_matcher/dict2.txt', 'rb') as h:
            self.d2 = [line.strip() for line in h.readlines()]
    
    def tearDown(self):
        pass
    
    def test_dictionary_matcher(self):
        """ Basic tests for dictionary matcher """
        D1 = ddlite_matcher.DictionaryMatch(dictionary=self.d1, label='Boat',
                                            match_attrib='lemmas')
        matches_1 = [[i for i,_ in D1.apply(s)] for s in self.sents]
        for i in range(len(matches_1)):
            if i == 1:
                self.assertEqual(matches_1[i], [[4], [9], [21], [24]])
            else:
                self.assertEqual(len(matches_1[i]), 0)
                
        D2 = ddlite_matcher.DictionaryMatch(dictionary=self.d2, label='PPL',
                                            match_attrib='lemmas')
        matches_2 = [[i for i,_ in D2.apply(s)] for s in self.sents]
        self.assertEqual(sorted(matches_2[64]), sorted([[3,4,5], [0,1]]))
        self.assertEqual(sorted(matches_2[1]), sorted([[30,31], [6], [28]]))

if __name__ == '__main__':
    unittest.main()
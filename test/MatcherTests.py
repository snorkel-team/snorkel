import os, sys, unittest

sys.path.insert(1, os.path.join(sys.path[0], '..'))
import ddlite_matcher
from ddlite_parser import SentenceParser

class TestMatchers(unittest.TestCase):
    
    def setUp(self):
        
        with open('test/data_matcher/cc_kurt.txt', 'rb') as f:
            self.txt = f.read()
        sp = SentenceParser()
        self.sents = list(sp.parse(self.txt))
        
        with open('test/data_matcher/dict1.txt', 'rb') as g:
            self.d1 = [line.strip() for line in g.readlines()]
        
        with open('test/data_matcher/dict2.txt', 'rb') as h:
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
        
    def test_regex_matcher(self):
        """ Basic tests for regex matcher """
        s = self.sents[64]
        
        R1 = ddlite_matcher.RegexFilterAll(label='Caps', regex_pattern=r'[A-Z]+',
                                           ignore_case=False)
        self.assertEqual(list(R1.apply(s)), [])
        self.assertEqual(len(list(R1._apply(s, idxs=[0,1,3,4,5]))[0][0]), 5)
        
        R2 = ddlite_matcher.RegexFilterAny(label='ACaps', regex_pattern=r'[A-Z]+',
                                           ignore_case=False)
        self.assertEqual(len(list(R2.apply(s))), 1)
        
        R3 = ddlite_matcher.RegexFilterConcat(label='NameName', regex_pattern=r'nn[a-z][0-9]nn[a-z]',
                                              match_attrib='poses', ignore_case=True, sep='8')
        self.assertEqual(len(list(R3.apply(s))), 1)
        
        R4 = ddlite_matcher.RegexNgramMatch(label='NameVb', regex_pattern=r'nn[a-z][0-9]vb[a-z]',
                                            match_attrib='poses', ignore_case=True, sep='8')  
        self.assertEqual(list(R4.apply(s)), [([1,2], 'NameVb')])

    def test_composition(self):
        s = self.sents[1]
        
        D_bt = ddlite_matcher.DictionaryMatch(dictionary=self.d1, label='Boat',
                                              match_attrib='lemmas')
        CE = ddlite_matcher.RegexNgramMatch(D_bt, label='root', regex_pattern=r'ROO+',
                                            match_attrib='dep_labels', ignore_case=False)
        self.assertEqual(list(CE.apply(s)), [([4], 'root : Boat')])
    
    def test_union(self):
        """ Test union operator """
        s = self.sents[1]
        
        D_ppl = ddlite_matcher.DictionaryMatch(dictionary=self.d2, label='PPL',
                                               match_attrib='lemmas')
        R_adj = ddlite_matcher.RegexNgramMatch(label='adj_nn', regex_pattern=r'jj[0-9]nn+',
                                               match_attrib='poses', ignore_case=True, sep='8')
        U = ddlite_matcher.Union(D_ppl, R_adj)
        self.assertEqual(sorted(list(U.apply(s))), sorted([([30, 31],'PPL'),
                                                           ([6],'PPL'),
                                                           ([28],'PPL'),
                                                           ([8, 9],'adj_nn')]))

if __name__ == '__main__':
    unittest.main()
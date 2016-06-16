import os, requests, sys, unittest

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ddlite_parser import *

class TestParsers(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.sp = SentenceParser()

    @classmethod
    def tearDownClass(cls):
        cls.sp._kill_pserver()

    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_doc_parser(self):
        """ Basic tests for document parsers """
        d1 = DocParser('test/data/*.txt', sp=self.sp)
        self.assertEqual(len(list(d1.readDocs())), 3)
        self.assertEqual(len(d1.parseDocSentences()), 205)
        
        d2 = DocParser('test/data/', sp=self.sp)
        self.assertEqual(len(list(d2.readDocs())), 4)
        
        d3 = DocParser('test/data/25075304.txt', sp=self.sp)
        self.assertEqual(len(list(d3.readDocs())), 1)
        
        d4 = DocParser('test/data/', HTMLReader(), sp=self.sp)
        self.assertEqual(len(list(d4.readDocs())), 1)
        sents = d4.parseDocSentences()
        self.assertEqual(len(list(sents)), 47)
    
    def test_sentence_parser(self):
        """ Basic tests for sentence parser """
        r = requests.get(self.sp.endpoint)
        self.assertEqual(r.status_code, requests.codes.ok)
        
        with open('test/data/25075304.txt', 'rb') as f:
            sents = list(self.sp.parse(f.read(), '25075304.txt'))
        self.assertEqual(len(list(sents)), 50)
        self.assertEqual(len(sents[0].words), 13)       


if __name__ == '__main__':
    unittest.main()

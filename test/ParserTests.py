import os, requests, sys, unittest

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ddlite_parser import *

class TestParsers(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_doc_parser(self):
        """ Basic tests for document parsers """
        d1 = DocParser('test/data/*.txt')
        self.assertEqual(len(list(d1.parseDocs())), 3)
        self.assertEqual(len(d1.parseDocSentences()), 3)
        
        d2 = DocParser('test/data/')
        self.assertEqual(len(list(d2.parseDocs())), 4)
        
        d3 = DocParser('test/data/25075304.txt')
        self.assertEqual(len(list(d3.parseDocs())), 1)
        
        d4 = DocParser('test/data/', HTMLParser())
        self.assertEqual(len(list(d4.parseDocs())), 1)
        self.assertEqual(len(list(d4.parseDocSentences()[0])), 47)
    
    def test_sentence_parser(self):
        """ Basic tests for sentence parser """
        sp = SentenceParser()
        r = requests.get(sp.endpoint)
        self.assertEqual(r.status_code, requests.codes.ok)
        
        with open('test/data/25075304.txt', 'rb') as f:
            sents = list(sp.parse(f.read()))
        self.assertEqual(len(list(sents)), 50)
        self.assertEqual(len(sents[0].words), 13)        

if __name__ == '__main__':
    unittest.main()
import unittest, requests
from parser import *

class TestParsers(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_doc_parser(self):
        d1 = DocParser('data/*.txt')
        self.assertEqual(len(d1.parse()), 3)
        
        d2 = DocParser('data/')
        self.assertEqual(len(d2.parse()), 4)
        
        d3 = DocParser('data/25075304.txt')
        self.assertEqual(len(d3.parse()), 1)
        
        d4 = DocParser('data/', HTMLParser())
        self.assertEqual(len(d4.parse()), 1)
        #self.assertEqual(len(d4.parseDocSentences()), ?)
    
    def test_sentence_parser(self):
        sp = SentenceParser()
        r = requests.get(sp.endpoint)
        self.assertEqual(r.status_code, requests.codes.ok)
        
        with open('data/25075304.txt', 'rb') as f:
            sents = list(sp.parse(f.read()))
        #self.assertEqual(len(sents), ?)
        #self.assertEqual(len(sents[0].words), ?)        

if __name__ == '__main__':
    unittest.main()
import os, sys, unittest
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from snorkel.parser import CorpusParser, HTMLParser, TableParser
from time import sleep

ROOT = os.environ['SNORKELHOME']

class TestTables(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.table_parser = TableParser()

    @classmethod
    def tearDownClass(cls):
        sleep(1)
        cls.table_parser._kill_pserver()

    def test_parsing(self):
        doc_parser = HTMLParser(path='tutorial/data/diseases/diseases.xhtml')
        cp = CorpusParser(doc_parser, self.table_parser)
        corpus = cp.parse_corpus(name='Test Corpus')
        self.assertEqual(len(corpus.documents), 1)
        self.assertEqual(len(corpus.documents[0].tables), 2)
        self.assertEqual(len(corpus.documents[0].cells), 24)
        self.assertEqual(len(corpus.documents[0].phrases), 25)
        self.assertEqual(corpus.documents[0].tables[0].cells[0].phrases[0].text, "Disease")

if __name__ == '__main__':
    unittest.main()

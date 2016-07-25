import os, requests, sys, unittest
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import cPickle
from snorkel.parser import *

ROOT = os.environ['SNORKELHOME']

class TestParsers(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sp = SentenceParser()

    @classmethod
    def tearDownClass(cls):
        cls.sp._kill_pserver()

    def test_parser_1(self):
        """Tests the XMLDocParser and SentenceParser subclasses"""

        # Load correct parses
        with open(ROOT + '/test/data/CDR_TestSet_docs.pkl', 'rb') as f:
            gold_docs = cPickle.load(f)

        with open(ROOT + '/test/data/CDR_TestSet_sents.pkl', 'rb') as f:
            gold_sents = cPickle.load(f)

        # Set up the doc parser
        xml_parser = XMLDocParser(
            path=ROOT + '/test/data/CDR_TestSet.xml',
            doc='.//document',
            text='.//passage/text/text()',
            id='.//id/text()',
            keep_xml_tree=False)

        sent_parser = SentenceParser()

        corpus = Corpus(xml_parser, sent_parser, max_docs=20)

        print len(corpus.get_docs())
        print len(corpus.get_contexts())

        self.assertEqual(corpus.get_docs(), gold_docs)
        self.assertEqual(corpus.get_contexts(), gold_sents)

if __name__ == '__main__':
    unittest.main()

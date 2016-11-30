import os, requests, sys, unittest
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import cPickle
from snorkel.parser import *
from snorkel.models import Corpus
from snorkel import SnorkelSession

os.environ['SNORKELHOME'] = '/home/pabajaj/snorkel'
DATA_PATH = os.environ['SNORKELHOME']+'/test/data/table_test/unit_tests/'
class TestVisualParsers(unittest.TestCase):
    
    @classmethod
    def setUpClass(self):
        from snorkel import SnorkelSession
        self.session = SnorkelSession()
        #os.remove('snorkel.db')
        
        self.doc_parser = HTMLParser(DATA_PATH)
        self.sent_parser = OmniParser(pdf_path=DATA_PATH, blacklist=["style", "ul"], flatten=["span"], visual=True, session=self.session)
        
        self.cp = CorpusParser(self.doc_parser, self.sent_parser, max_docs=2)
        self.corpus = self.cp.parse_corpus(name='Hardware', session=self.session)
        self.doc = self.corpus.documents[0]
        self.doc2 = self.corpus.documents[1]
        if(self.doc.name == "doc_unit_test"):
            new_doc = self.doc
            self.doc = self.doc2
            self.doc2 = new_doc
        print "Testing on files:", self.doc.name, self.doc2.name
    
    @classmethod
    def tearDownClass(self):
        pass
    
    def test_phrase_page_and_coordinates(self):
        """Tests if all words in all phrases have page numbers and coordinates"""
        for phrase in self.doc.phrases:
            self.assertTrue(None not in phrase.page)
            self.assertTrue(None not in phrase.top)
            self.assertTrue(None not in phrase.left)
            self.assertTrue(None not in phrase.right)
            self.assertTrue(None not in phrase.bottom)
    
    def test_num_phrases(self):    
        """Tests if the number of phrases are the same after parsing"""
        self.assertTrue(len(self.doc.phrases)==123)
    
    def test_flatten_entry_tags_handled(self):
        """Test if text within tags with flatten entry is parsed correctly - phrases are not split by flatten tags"""
        #check a phrase with span tag gets parsed correctly
        phrase_last = self.doc2.phrases[-2]
        self.assertTrue("OutSpan" in phrase_last.words and "InSpan" in phrase_last.words)
    
    def test_blacklist_tag_not_parsed(self):
        """Test for checking that content in BlackList doesn't appear in Phrases"""
        
        #check that first phrase is not style phrase
        phrase = self.doc2.phrases[0]
        self.assertTrue("margin" not in phrase.words)

        #check that the 'ul' tag was not parsed
        phrase = self.doc2.phrases[-2]
        self.assertTrue("False!" not in phrase.words)
    
    def test_all_rows_cols_filled_table(self):
        """Test if all rows and columns are filled"""

        row_max = 0
        col_max = 0
        table_grid = defaultdict(int)
        for phrase in self.doc2.tables[0].phrases:
            if(phrase.row_start != None):   # condition to not check caption of table
                for row in range(phrase.row_start, phrase.row_end+1):
                    row_max = max(row_max, phrase.row_end)
                    for col in range(phrase.col_start, phrase.col_end+1):
                        col_max = max(col_max, phrase.col_end)
                        table_grid[(row, col)] = table_grid[(row, col)] + 1
        for row in range(0, row_max+1):
            for col in range(0, col_max+1):
                self.assertTrue(table_grid[(row, col)] == 1)
    
    def test_nested_tables_parsed_correctly(self):
        """Test if phrases are correctly parsed in nested tables"""

        nested_table_phrases = self.doc2.tables[3].phrases
        self.assertTrue('A11' in nested_table_phrases[0].words)
        self.assertTrue('A12' in nested_table_phrases[1].words)
        self.assertTrue('A21' in nested_table_phrases[2].words)
        self.assertTrue('A22' in nested_table_phrases[3].words)

        parent_table_phrases = self.doc2.tables[2].phrases
        self.assertTrue('Nested' in parent_table_phrases[4].words)
        self.assertTrue('Nested' in parent_table_phrases[5].words)
    
    def test_attributes_match_manual_solution(self):
        """Test if attributes are correctly parsed at top, middle and bottom for a table"""

        #for tag at 'bottom' of table - that is phrases inside <p> tags; cells have "None" values in html_tag and html_attrs
        table = self.doc2.tables[1]
        self.assertTrue(table.phrases[0].html_tag=="p" and str(table.phrases[0].html_attrs)=="[('name', '1.1.p1')]")
        self.assertTrue(table.phrases[1].html_tag=="p" and str(table.phrases[1].html_attrs)=="[('name', '1.1.p2')]")
        self.assertTrue(table.phrases[2].html_tag=="p" and str(table.phrases[2].html_attrs)=="[('name', '1.2.p1')]")
        self.assertTrue(table.phrases[3].html_tag=="p" and str(table.phrases[3].html_attrs)=="[('name', '1.2.p2')]")
        self.assertTrue(table.phrases[4].html_tag=="p" and str(table.phrases[4].html_attrs)=="[('name', '2.1.p1')]")
        self.assertTrue(table.phrases[5].html_tag=="p" and str(table.phrases[5].html_attrs)=="[('name', '2.1.p2')]")
        self.assertTrue(table.phrases[6].html_tag=="p" and str(table.phrases[6].html_attrs)=="[('name', '2.2.p1')]")
        self.assertTrue(table.phrases[7].html_tag=="p" and str(table.phrases[7].html_attrs)=="[('name', '2.2.p2')]")
        #self.assertTrue(len(phrase.html_attrs)==0)
       
       
if __name__ == '__main__':
    unittest.main()

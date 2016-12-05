import cPickle
import os
import sys
import unittest

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from snorkel.visual import VisualLinker
from snorkel.utils_visual import get_pdf_dim

ROOT = os.environ['SNORKELHOME']
DATA_PATH = os.path.join(ROOT, 'test/data/table_test/')


class TestVisualFeatures(unittest.TestCase):
    def extract_pdf_words(self):
        with open(os.path.join(DATA_PATH, 'coord_map.pkl'), 'rb') as f:
            coordinate_map = cPickle.load(f)
        with open(os.path.join(DATA_PATH, 'pdf_words.pkl'), 'rb') as f:
            pdf_words_list = cPickle.load(f)
        viz = VisualLinker(None, None)
        # get visual features on a file with extension .pdf
        viz.pdf_file = DATA_PATH + '112823.pdf'
        viz.extract_pdf_words()
        self.assertEqual(pdf_words_list, viz.pdf_word_list)
        self.assertEqual(coordinate_map, viz.coordinate_map)

    def test_create_pdf(self):
        viz = VisualLinker(None, None)
        document_name = 'BC337-D'
        viz.pdf_path = DATA_PATH
        with open(os.path.join(DATA_PATH, 'html_string.pkl'), 'rb') as f:
            text = cPickle.load(f)
        viz.create_pdf(document_name, text)
        self.assertTrue(os.path.isfile(viz.pdf_path + document_name + '.pdf'))
        os.system('rm {}'.format(viz.pdf_path + document_name + '.pdf'))

    def test_get_pdf_dim(self):
        pdf_file = DATA_PATH + '112823.pdf'
        pdf_dim = get_pdf_dim(pdf_file)
        self.assertEqual(pdf_dim, (612, 792))

    def test_linking_accuracy(self):
        # confirm that linking accuracy is above some reasonable threshold (70%?)
        pass

if __name__ == '__main__':
    unittest.main()

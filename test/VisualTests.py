import cPickle
import os
import sys
import unittest

sys.path.append(os.environ['SNORKELHOME'] + 'snorkel/')
from visual import VisualLinker

DATA_PATH = os.environ['SNORKELHOME'] + 'test/data/table_test/'


class TestVisualFeatures(unittest.TestCase):
    def test_get_visual_features_PDF_extension(self):
        with open(os.path.join(DATA_PATH, 'coord_map.pkl'), 'rb') as f:
            coordinate_map = cPickle.load(f)
        with open(os.path.join(DATA_PATH, 'pdf_words.pkl'), 'rb') as f:
            pdf_words_list = cPickle.load(f)
        # get visual features on a file with extension .PDF
        viz = VisualLinker(None, None)
        viz.pdf_file = DATA_PATH + '112823.PDF'
        viz.extract_pdf_words()
        self.assertEqual(pdf_words_list, viz.pdf_word_list)
        self.assertEqual(coordinate_map, viz.coordinate_map)

    def test_get_visual_features_pdf_extension(self):
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

    def test_get_visual_features_from_html(self):
        # get visual features on a file with no PDF (generate one from html)
        pass

    def test_linking_accuracy(self):
        # confirm that linking accuracy is above some reasonable threshold (70%?)
        pass

    def test_create_pdf(self):
        viz = VisualLinker(None, None)
        document_name = 'BC337-D'
        viz.pdf_path = DATA_PATH
        with open(os.path.join(DATA_PATH, 'html_string.pkl'), 'rb') as f:
            text = cPickle.load(f)
        viz.create_pdf(document_name, text)
        self.assertTrue(os.path.isfile(viz.pdf_path + document_name + '.pdf'))
        os.system('rm {}'.format(viz.pdf_path + document_name + '.pdf'))

if __name__ == '__main__':
    unittest.main()

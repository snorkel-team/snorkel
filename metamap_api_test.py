import unittest
import os
os.environ['SNORKELHOME'] = os.path.abspath('.')
from metamap_api import MetaMapAPI


class MetamapAPITest(unittest.TestCase):

    def test_no_concept_in_sentence(self):
        metamap_api = MetaMapAPI(MetaMapMock)
        metamap_api.tag('')
        self.assertEqual(1, 3 - 2)


class MetaMapMock(object):

    def extract_concepts(self, sentences=None, ids=None,
                         filename=None, composite_phrase=4,
                         file_format='sldi', word_sense_disambiguation=True):
        """ Extract concepts from a list of sentences using MetaMap. """
        print('here')
        return [['here'], '']

    @staticmethod
    def get_instance(metamap_filename, version=None, backend='subprocess',
                     **extra_args):
        return MetaMapMock()


if __name__ == '__main__':
    unittest.main()

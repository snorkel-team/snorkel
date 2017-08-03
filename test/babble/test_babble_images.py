import os
import unittest

from snorkel import SnorkelSession
from snorkel.models import candidate_subclass
import image_explanations

class TestBabbleImages(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create initial snorkel.db
        # session = SnorkelSession()
        # candidate_subclass = candidate_subclass('Spouse', ['person1', 'person2'])
        # cls.sp = SemanticParser(candidate_subclass,  beam_width=10, top_k=-1)
        print("Set me up!")

    @classmethod
    def tearDownClass(cls):
        pass

    def test_boxes(self):
        # self.check_explanations(unittest_explanations.boxes)
        print("Hello world!")
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
import os
import unittest

from snorkel import SnorkelSession
from snorkel.models import candidate_subclass
from snorkel.contrib.babble import SemanticParser

from test_babble_base import TestBabbleBase
import image_explanations

class TestBabbleImage(TestBabbleBase):

    @classmethod
    def setUpClass(cls):
        cls.sp = SemanticParser(mode='image')

    def test_edges(self):
        self.check_explanations(image_explanations.edges)

    def test_corners(self):
        self.check_explanations(image_explanations.points)
 
    def test_boxes(self):
        self.check_explanations(image_explanations.boxes)

    def test_comparisons(self):
        self.check_explanations(image_explanations.comparisons)

    def test_quantified(self):
        self.check_explanations(image_explanations.quantified)
        
    def test_parser(self):
        self.check_explanations(image_explanations.parser)

suite = unittest.TestLoader().loadTestsFromTestCase(TestBabbleImage)
unittest.TextTestRunner(verbosity=2).run(suite)
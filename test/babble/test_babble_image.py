import os
import unittest

from snorkel import SnorkelSession
from snorkel.models import candidate_subclass
from snorkel.contrib.babble import SemanticParser
import image_explanations

class TestBabbleImage(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sp = SemanticParser(mode='image', beam_width=10, top_k=-1)

    @classmethod
    def tearDownClass(cls):
        pass
    
    def check_explanations(self, explanations):
        self.assertTrue(len(explanations))
        for e in explanations:
            LF_dict = self.sp.parse_and_evaluate(e, show_erroring=True) # show_nothing=True
            if not len(LF_dict['correct']) + len(LF_dict['passing']) > 0:
                print(LF_dict)
                self.sp.grammar.print_chart()
                parses = self.sp.parse(e, return_parses=True)
                import pdb; pdb.set_trace()
            # parses = self.sp.parse(e, return_parses=True)
            if e.semantics:
                self.assertTrue(len(LF_dict['correct']) > 0)
            else:
                self.assertTrue(len(LF_dict['passing']) > 0)
            self.assertTrue(len(LF_dict['correct']) + len(LF_dict['passing']) <= 3)

    def test_edges(self):
        self.check_explanations(image_explanations.edges)

    def test_corners(self):
        self.check_explanations(image_explanations.points)
 
    def test_comparisons(self):
        self.check_explanations(image_explanations.comparisons)

suite = unittest.TestLoader().loadTestsFromTestCase(TestBabbleImage)
unittest.TextTestRunner(verbosity=2).run(suite)
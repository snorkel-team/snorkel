import os
import unittest

from snorkel.contrib.babble import SemanticParser

from test_babble_base import TestBabbleBase
import core_explanations

class TestBabbleCore(TestBabbleBase):

    @classmethod
    def setUpClass(cls):
        cls.sp = SemanticParser(candidate_class=None, 
                                user_lists=core_explanations.get_user_lists(), 
                                beam_width=10, 
                                top_k=-1)

    def test_logic(self):
        self.check_explanations(core_explanations.logic)

    def test_grouping(self):
        self.check_explanations(core_explanations.grouping)

    def test_integers(self):
        self.check_explanations(core_explanations.integers)

    def test_lists(self):
        self.check_explanations(core_explanations.lists)

    def test_absorption(self):
        self.check_explanations(core_explanations.absorption)

    def test_translate(self):
        semantics = ('.root', ('.label', ('.bool', True), ('.and', ('.bool', True), ('.bool', True))))
        pseudocode = 'return 1 if (True and True) else 0'
        self.assertEqual(self.sp.translate(semantics), pseudocode)


suite = unittest.TestLoader().loadTestsFromTestCase(TestBabbleCore)
unittest.TextTestRunner(verbosity=2).run(suite)

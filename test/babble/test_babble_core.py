import os
import unittest

from snorkel.contrib.babble import SemanticParser
import core_explanations

class TestBabbleCore(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sp = SemanticParser(candidate_class=None, 
                                user_lists=core_explanations.get_user_lists(), 
                                beam_width=10, 
                                top_k=-1)

    @classmethod
    def tearDownClass(cls):
        pass

    def check_explanations(self, explanations):
        self.assertTrue(len(explanations))
        for e in explanations:
            LF_dict = self.sp.parse_and_evaluate(e, show_erroring=True)
            if e.semantics:
                self.assertTrue(len(LF_dict['correct']) > 0)
            else:
                self.assertTrue(len(LF_dict['passing']) > 0)
            self.assertTrue(len(LF_dict['correct']) + len(LF_dict['passing']) <= 3)


    def test_logic(self):
        self.check_explanations(core_explanations.logic)

    def test_grouping(self):
        self.check_explanations(core_explanations.grouping)

    def test_integers(self):
        self.check_explanations(core_explanations.integers)

    def test_absorption(self):
        self.check_explanations(core_explanations.absorption)

    def test_translate(self):
        semantics = ('.root', ('.label', ('.bool', True), ('.and', ('.bool', True), ('.bool', True))))
        pseudocode = 'return 1 if (True and True) else 0'
        self.assertEqual(self.sp.translate(semantics), pseudocode)


suite = unittest.TestLoader().loadTestsFromTestCase(TestBabbleCore)
unittest.TextTestRunner(verbosity=2).run(suite)

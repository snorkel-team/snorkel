import sys
import unittest

class TestBabbleBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def check_explanations(self, explanations):
        num_explanations = len(explanations)
        num_parses = 0
        self.assertTrue(len(explanations))
        for e in explanations:
            if e.candidate and not isinstance(e.candidate, tuple):
                e.candidate = self.candidate_hash[e.candidate]
            LF_dict = self.sp.parse_and_evaluate(e, show_erroring=True)
            num_correct = len(LF_dict['correct'])
            num_passing = len(LF_dict['passing'])
            num_acceptable = num_correct + num_passing
            if not num_acceptable > 0:
                print(LF_dict)
                self.sp.grammar.print_chart()
                parses = self.sp.parse(e, return_parses=True)
                import pdb; pdb.set_trace()
            if e.semantics:
                self.assertTrue(num_correct > 0)
            else:
                self.assertTrue(num_passing > 0)
            self.assertTrue(num_acceptable <= 3)
            num_parses += num_acceptable
        sys.stdout.write("{}/{} ({}%) - ".format(num_parses, num_explanations, 
            float(num_parses)/num_explanations * 100))
        sys.stdout.flush()
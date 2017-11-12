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
        for exp in explanations:
            if exp.candidate and not isinstance(exp.candidate, tuple):
                exp.candidate = self.candidate_map[exp.candidate]
            parse_dict = self.sp.parse_and_evaluate(exp, show_erroring=True)
            # TEMP: Use for getting semantics to put in Explanation.semantics
            # parses = self.sp.parse(exp, return_parses=True)
            # print(parses[0].semantics)
            # TEMP
            num_correct = len(parse_dict['correct'])
            num_passing = len(parse_dict['passing'])
            num_failing = len(parse_dict['failing'])
            num_erroring = len(parse_dict['erroring'])
            num_acceptable = num_correct + num_passing
            if not num_acceptable > 0:
                print(parse_dict)
                if num_failing:
                    for failing in parse_dict['failing']:
                        print("Failed parse:")
                        print(self.sp.grammar.translate(failing.semantics))
                if num_erroring:
                    print("It should not be possible to parse a function that throws an error:")
                    self.sp.grammar.print_chart()
                    parses = self.sp.parse(exp, return_parses=True)
                    import pdb; pdb.set_trace()
            if exp.semantics:
                self.assertTrue(num_correct > 0)
            else:
                self.assertTrue(num_passing > 0)
            self.assertTrue(num_acceptable <= 3)
            num_parses += num_acceptable
        sys.stdout.write("{}/{} ({}%) - ".format(num_parses, num_explanations, 
            float(num_parses)/num_explanations * 100))
        sys.stdout.flush()
import math
from numbskull.inference import FACTORS
from scipy import sparse
from snorkel.learning.gen_learning import GenerativeModel, DEP_EXCLUSIVE, DEP_REINFORCING, DEP_FIXING, DEP_SIMILAR
import unittest
import random


class TestSupervised(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_supervised(self):
        # Defines a label matrix
        n = 1000
        L = sparse.lil_matrix((n, 6))

        for i in range(n):
            y = 2 * random.randint(0, 1) - 1
            # First four LFs always vote, and have decent acc
            L[i, 0] = y * (2 * (random.random() < 0.75) - 1)
            L[i, 1] = y * (2 * (random.random() < 0.75) - 1)
            L[i, 2] = y * (2 * (random.random() < 0.75) - 1)
            L[i, 3] = y * (2 * (random.random() < 0.75) - 1)

            # The fifth LF is very accurate and essentially corrects the first two
            if L[i, 0] != y and L[i, 1] != y:
                L[i, 4] = y * (2 * (random.random() < 0.9) - 1)

            # The sixth LF is a small supervised set
            # Random 5% are labeled, along with things the correcting LF marked
            if random.random() < 0.05 or L[i, 4] != 0:
                L[i, 5] = y

        # Test with priors
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(L, LF_priors=[0.75, 0.75, 0.75, 0.75, 0.9, 0.99], is_fixed=[False, False, False, False, False, True], reg_type=2, reg_param=1, epochs=0)
        print(gen_model.weights.lf_accuracy())
        print(gen_model.weights.lf_propensity)

        gen_model.train(L, LF_priors=[0.75, 0.75, 0.75, 0.75, 0.9, 0.99], is_fixed=[False, False, False, False, False, True], reg_type=2, reg_param=1)
        print(gen_model.weights.lf_accuracy())
        print(gen_model.weights.lf_propensity)

        # Test without supervised
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(L, reg_type=0)
        print(gen_model.weights.lf_accuracy())
        print(gen_model.weights.lf_propensity)

        # Test with supervised
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(L, LF_priors=[0.5, 0.5, 0.5, 0.5, 0.5, 0.99], is_fixed=[False, False, False, False, False, True], reg_type=0)
        print(gen_model.weights.lf_accuracy())
        print(gen_model.weights.lf_propensity)

if __name__ == '__main__':
    unittest.main()

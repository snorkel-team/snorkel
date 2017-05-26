import math
from numbskull.inference import FACTORS
from scipy import sparse
from snorkel.learning.gen_learning import GenerativeModel, DEP_EXCLUSIVE, DEP_REINFORCING, DEP_FIXING, DEP_SIMILAR
import unittest
import random
import numpy as np


class TestSupervised(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_supervised(self):
        # A set of true priors
        LF_priors = [0.75, 0.75, 0.75, 0.75, 0.9]
        label_prior = 0.99

        # Defines a label matrix
        n = 1000
        L = sparse.lil_matrix((n, 5))

        # Store the supervised gold labels separately
        labels = np.zeros(n)

        for i in range(n):
            y = 2 * random.randint(0, 1) - 1
            # First four LFs always vote, and have decent acc
            L[i, 0] = y * (2 * (random.random() < LF_priors[0]) - 1)
            L[i, 1] = y * (2 * (random.random() < LF_priors[1]) - 1)
            L[i, 2] = y * (2 * (random.random() < LF_priors[2]) - 1)
            L[i, 3] = y * (2 * (random.random() < LF_priors[3]) - 1)

            # The fifth LF is very accurate and essentially corrects the first two
            if L[i, 0] != y and L[i, 1] != y:
                L[i, 4] = y * (2 * (random.random() < LF_priors[4]) - 1)

            # The sixth LF is a small supervised set
            # Random 5% are labeled, along with things the correcting LF marked
            if random.random() < 0.05 or L[i, 4] != 0:
                labels[i] = y

        # Test with priors -- first check init vals are correct
        print("Testing init:")
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(
            L,
            LF_priors=LF_priors,
            labels=labels,
            label_prior=label_prior,
            reg_type=2,
            reg_param=1,
            epochs=0
        )
        accs = gen_model.weights.lf_accuracy()
        priors = np.array(LF_priors + [label_prior])
        print(accs)
        print(gen_model.weights.lf_propensity)
        self.assertTrue(np.linalg.norm(accs - priors) < 1e-5)

        # Now test that estimated LF accs are not too far off
        tol = 0.1
        print("\nTesting estimated LF accs (TOL=%s)" % tol)
        gen_model.train(
            L,
            LF_priors=LF_priors,
            labels=labels,
            label_prior=label_prior,
            reg_type=2,
            reg_param=1
        )
        accs = gen_model.weights.lf_accuracy()
        priors = np.array(LF_priors + [label_prior])
        print(accs)
        print(np.abs(accs - priors))
        print(gen_model.weights.lf_propensity)
        self.assertTrue(np.all(np.abs(accs - priors) < tol))

        # Test without supervised
        print("\nTesting without supervised")
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(L, reg_type=0)
        accs = gen_model.weights.lf_accuracy()
        priors = np.array(LF_priors)
        print(accs)
        print(np.abs(accs - priors))
        print(gen_model.weights.lf_propensity)
        # self.assertTrue(np.all(np.abs(accs - priors) < tol))

        # Test with supervised
        print("\nTesting with supervised, without priors")
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(
            L,
            labels=labels,
            label_prior=label_prior,
            reg_type=0
        )
        accs = gen_model.weights.lf_accuracy()
        priors = np.array(LF_priors + [label_prior])
        print(accs)
        print(np.abs(accs - priors))
        print(gen_model.weights.lf_propensity)
        self.assertTrue(accs[4] > 0.6)

if __name__ == '__main__':
    unittest.main()

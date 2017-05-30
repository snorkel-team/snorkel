import math
from numbskull.inference import FACTORS
from scipy import sparse
from snorkel.learning.gen_learning import GenerativeModel, DEP_EXCLUSIVE, DEP_REINFORCING, DEP_FIXING, DEP_SIMILAR
import unittest
import random
import numpy as np


class TestCategorical(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_categorical(self):
        # A set of true priors
        tol = 0.2
        LF_acc_priors = [0.75, 0.75, 0.75, 0.75, 0.9]
        label_prior = 0.999
        cardinality = 4

        def get_lf(label, cardinality, acc):
            if random.random() < acc:
                return label + 1
            lf = random.randint(0, cardinality - 1)
            if (lf >= label):
                lf += 1
            return lf + 1

        # Defines a label matrix
        n = 10000
        L = sparse.lil_matrix((n, 5), dtype=np.int64)

        # Store the supervised gold labels separately
        labels = np.zeros(n)

        for i in range(n):
            y = random.randint(0, cardinality)
            # First four LFs always vote, and have decent acc
            L[i, 0] = get_lf(y, cardinality, LF_acc_priors[0])
            L[i, 1] = get_lf(y, cardinality, LF_acc_priors[1])
            L[i, 2] = get_lf(y, cardinality, LF_acc_priors[2])
            L[i, 3] = get_lf(y, cardinality, LF_acc_priors[3])

            # The fifth LF is very accurate but has a much smaller coverage
            if random.random() < 0.2:
                L[i, 4] = get_lf(y, cardinality, LF_acc_priors[4])

            # The sixth LF is a small supervised set
            if random.random() < 0.1:
                labels[i] = y + 1

        # Test with priors -- first check init vals are correct
        print("Testing init:")
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(
            L,
            LF_acc_priors=LF_acc_priors,
            labels=labels,
            label_prior=label_prior,
            reg_type=2,
            reg_param=1,
            epochs=0
        )
        stats = gen_model.learned_lf_stats()
        accs = stats["Accuracy"]
        print(accs)
        print(gen_model.weights.lf_propensity)
        priors = np.array(LF_acc_priors + [label_prior])
        self.assertTrue(np.all(np.abs(accs - priors) < tol))

        # Now test that estimated LF accs are not too far off
        print("\nTesting estimated LF accs (TOL=%s)" % tol)
        gen_model.train(
            L,
            LF_acc_priors=LF_acc_priors,
            labels=labels,
            label_prior=label_prior,
            reg_type=0,
            reg_param=0.0
        )
        stats = gen_model.learned_lf_stats()
        accs = stats["Accuracy"]
        coverage = stats["Coverage"]
        print(accs)
        print(coverage)
        priors = np.array(LF_acc_priors + [label_prior])
        self.assertTrue(np.all(np.abs(accs - priors) < tol))
        self.assertTrue(np.all(np.abs(coverage - np.array([1, 1, 1, 1, 0.2, 0.1]) < tol)))

        # Test without supervised
        print("\nTesting without supervised")
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(L, reg_type=0)
        stats = gen_model.learned_lf_stats()
        accs = stats["Accuracy"]
        coverage = stats["Coverage"]
        print(accs)
        print(coverage)
        priors = np.array(LF_acc_priors)
        self.assertTrue(np.all(np.abs(accs - priors) < tol))
        self.assertTrue(np.all(np.abs(coverage - np.array([1, 1, 1, 1, 0.2]) < tol)))

        # Test with supervised
        print("\nTesting with supervised, without priors")
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(
            L,
            labels=labels,
            label_prior=label_prior,
            reg_type=0
        )
        stats = gen_model.learned_lf_stats()
        accs = stats["Accuracy"]
        coverage = stats["Coverage"]
        print(accs)
        print(coverage)
        priors = np.array(LF_acc_priors + [label_prior])
        self.assertTrue(np.all(np.abs(accs - priors) < tol))
        self.assertTrue(np.all(np.abs(coverage - np.array([1, 1, 1, 1, 0.2, 0.1]) < tol)))

        # Test without supervised, and (intentionally) bad priors, but weak strength
        print("\nTesting without supervised, with bad priors (weak)")
        gen_model = GenerativeModel(lf_propensity=True)
        bad_prior = [0.9, 0.8, 0.7, 0.6, 0.5]
        gen_model.train(
            L,
            LF_acc_priors=bad_prior,
            reg_type=0
        )
        stats = gen_model.learned_lf_stats()
        accs = stats["Accuracy"]
        coverage = stats["Coverage"]
        print(accs)
        print(coverage)
        priors = np.array(LF_acc_priors)
        self.assertTrue(np.all(np.abs(accs - priors) < tol))

        # Test without supervised, and (intentionally) bad priors
        print("\nTesting without supervised, with bad priors (strong)")
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(
            L,
            LF_acc_priors=bad_prior,
            reg_type=2,
            reg_param=100 * n
        )
        stats = gen_model.learned_lf_stats()
        accs = stats["Accuracy"]
        coverage = stats["Coverage"]
        print(accs)
        self.assertTrue(np.all(np.abs(accs - np.array(bad_prior)) < tol))

if __name__ == '__main__':
    unittest.main()

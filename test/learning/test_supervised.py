from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import math
from numbskull.inference import FACTORS
from scipy import sparse
from snorkel.learning.gen_learning import (
    GenerativeModel, DEP_EXCLUSIVE, DEP_REINFORCING, DEP_FIXING, DEP_SIMILAR
)
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
        tol = 0.1
        LF_acc_priors = [0.75, 0.75, 0.75, 0.75, 0.9]
        cardinality = 2
        LF_acc_prior_weights = [0.5 * np.log((cardinality - 1.0) * x / (1 - x)) for x in LF_acc_priors]
        label_prior = 1

        # Defines a label matrix
        n = 10000
        L = sparse.lil_matrix((n, 5), dtype=np.int64)

        # Store the supervised gold labels separately
        labels = np.zeros(n, np.int64)

        for i in range(n):
            y = 2 * random.randint(0, 1) - 1
            # First four LFs always vote, and have decent acc
            L[i, 0] = y * (2 * (random.random() < LF_acc_priors[0]) - 1)
            L[i, 1] = y * (2 * (random.random() < LF_acc_priors[1]) - 1)
            L[i, 2] = y * (2 * (random.random() < LF_acc_priors[2]) - 1)
            L[i, 3] = y * (2 * (random.random() < LF_acc_priors[3]) - 1)

            # The fifth LF is very accurate but has a much smaller coverage
            if random.random() < 0.2:
                L[i, 4] = y * (2 * (random.random() < LF_acc_priors[4]) - 1)

            # The sixth LF is a small supervised set
            if random.random() < 0.1:
                labels[i] = y

        # Test with priors -- first check init vals are correct
        print("Testing init:")
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(
            L,
            LF_acc_prior_weights=LF_acc_prior_weights,
            labels=labels,
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
            LF_acc_prior_weights=LF_acc_prior_weights,
            labels=labels,
            reg_type=0,
            reg_param=0.0,
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
        bad_prior_weights = [0.5 * np.log((cardinality - 1.0) * x / (1 - x)) for x in bad_prior]
        gen_model.train(
            L,
            LF_acc_prior_weights=bad_prior_weights,
            reg_type=0,
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
            LF_acc_prior_weights=bad_prior_weights,
            reg_type=2,
            reg_param=100 * n,
        )
        stats = gen_model.learned_lf_stats()
        accs = stats["Accuracy"]
        coverage = stats["Coverage"]
        print(accs)
        self.assertTrue(np.all(np.abs(accs - np.array(bad_prior)) < tol))

if __name__ == '__main__':
    unittest.main()

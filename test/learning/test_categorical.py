from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import math
from numbskull.inference import FACTORS
from scipy import sparse
from snorkel.learning.gen_learning import GenerativeModel, DEP_EXCLUSIVE, DEP_REINFORCING, DEP_FIXING, DEP_SIMILAR
import unittest
import random
import numpy as np
from time import time


class TestCategorical(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def _generate_L(self, LF_acc_priors, cardinality=4, n=10000):
        """
        Generate label matrix and ground truth labels given LF acc prior 
        probabilities, a fixed cardinality, and number of candidates.

        Returns a CSR_sparse label matrix L and supervised labels array.
        """
        def get_lf(label, cardinality, acc):
            if random.random() < acc:
                return label + 1
            lf = random.randint(0, cardinality - 2)
            if (lf >= label):
                lf += 1
            return lf + 1

        # Defines a label matrix
        L = sparse.lil_matrix((n, 5), dtype=np.int64)

        # Store the supervised gold labels separately
        labels = np.zeros(n, np.int64)

        for i in range(n):
            y = random.randint(0, cardinality - 1)
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

        # Return as CSR sparse matrix
        return sparse.csr_matrix(L), labels

    def _generate_L_scoped_categorical(self, LF_acc_priors,
        per_candidate_cardinality=4, full_cardinality=4, n=10000):
        """
        Generate label matrix and ground truth labels given LF acc prior 
        probabilities, a "full" cardinality of the problem, and a per-candidate
        cardinality.

        Returns a CSR_sparse label matrix L and supervised labels array. 
        Generates and returns a random set of candidate_ranges as well.
        """
        L = sparse.lil_matrix((n, 5), dtype=np.int64)
        labels = np.zeros(n, np.int64)
        candidate_ranges = []
        for i in range(n):
            # Generate a random support set
            c_range = list(range(1, full_cardinality + 1))
            np.random.shuffle(c_range)
            c_range = c_range[:per_candidate_cardinality]
            candidate_ranges.append(c_range)

            # Generate a true label
            y = c_range[random.randint(0, per_candidate_cardinality - 1)]

            # Generate the labels same as in self._generate_L
            for j in range(5):
                # LF 5 has smaller coverage
                if j == 4 and random.random() > 0.2:
                    continue
                label = y
                # Some probability of being incorrect
                if random.random() > LF_acc_priors[j]:
                    while label == y:
                        label = c_range[
                            random.randint(0, per_candidate_cardinality - 1)]
                L[i, j] = label

            # Small supervised training set
            if random.random() < 0.1:
                labels[i] = y

        # Return as CSR sparse matrix
        return sparse.csr_matrix(L), labels, candidate_ranges

    def _test_categorical(self, L, LF_acc_priors, labels, label_prior=1, 
        candidate_ranges=None, cardinality=4, tol=0.1, n=10000):
        """Run a suite of tests."""
        # Map to log scale weights
        LF_acc_prior_weights = [0.5 * np.log((cardinality - 1.0) * x / (1 - x)) for x in LF_acc_priors]

        # Test with priors -- first check init vals are correct
        print("Testing init:")
        t0 = time()
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(
            L,
            LF_acc_prior_weights=LF_acc_prior_weights,
            labels=labels,
            reg_type=2,
            reg_param=1,
            epochs=0,
            candidate_ranges=candidate_ranges
        )
        stats = gen_model.learned_lf_stats()
        accs = stats["Accuracy"]
        print(accs)
        print(gen_model.weights.lf_propensity)
        priors = np.array(LF_acc_priors + [label_prior])
        self.assertTrue(np.all(np.abs(accs - priors) < tol))
        print("Finished in {0} sec.".format(time()-t0))

        # Now test that estimated LF accs are not too far off
        print("\nTesting estimated LF accs (TOL=%s)" % tol)
        t0 = time()
        gen_model.train(
            L,
            LF_acc_prior_weights=LF_acc_prior_weights,
            labels=labels,
            reg_type=0,
            reg_param=0.0,
            candidate_ranges=candidate_ranges
        )
        stats = gen_model.learned_lf_stats()
        accs = stats["Accuracy"]
        coverage = stats["Coverage"]
        print(accs)
        print(coverage)
        priors = np.array(LF_acc_priors + [label_prior])
        self.assertTrue(np.all(np.abs(accs - priors) < tol))
        self.assertTrue(np.all(np.abs(coverage - np.array([1, 1, 1, 1, 0.2, 0.1]) < tol)))
        print("Finished in {0} sec.".format(time()-t0))

        # Test without supervised
        print("\nTesting without supervised")
        t0 = time()
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(L, reg_type=0, candidate_ranges=candidate_ranges)
        stats = gen_model.learned_lf_stats()
        accs = stats["Accuracy"]
        coverage = stats["Coverage"]
        print(accs)
        print(coverage)
        priors = np.array(LF_acc_priors)
        self.assertTrue(np.all(np.abs(accs - priors) < tol))
        self.assertTrue(np.all(np.abs(coverage - np.array([1, 1, 1, 1, 0.2]) < tol)))
        print("Finished in {0} sec.".format(time()-t0))

        # Test with supervised
        print("\nTesting with supervised, without priors")
        t0 = time()
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(
            L,
            labels=labels,
            reg_type=0,
            candidate_ranges=candidate_ranges
        )
        stats = gen_model.learned_lf_stats()
        accs = stats["Accuracy"]
        coverage = stats["Coverage"]
        print(accs)
        print(coverage)
        priors = np.array(LF_acc_priors + [label_prior])
        self.assertTrue(np.all(np.abs(accs - priors) < tol))
        self.assertTrue(np.all(np.abs(coverage - np.array([1, 1, 1, 1, 0.2, 0.1]) < tol)))
        print("Finished in {0} sec.".format(time()-t0))

        # Test without supervised, and (intentionally) bad priors, but weak strength
        print("\nTesting without supervised, with bad priors (weak)")
        t0 = time()
        gen_model = GenerativeModel(lf_propensity=True)
        bad_prior = [0.9, 0.8, 0.7, 0.6, 0.5]
        bad_prior_weights = [0.5 * np.log((cardinality - 1.0) * x / (1 - x)) for x in bad_prior]
        gen_model.train(
            L,
            LF_acc_prior_weights=bad_prior_weights,
            reg_type=0,
            candidate_ranges=candidate_ranges
        )
        stats = gen_model.learned_lf_stats()
        accs = stats["Accuracy"]
        coverage = stats["Coverage"]
        print(accs)
        print(coverage)
        priors = np.array(LF_acc_priors)
        self.assertTrue(np.all(np.abs(accs - priors) < tol))
        print("Finished in {0} sec.".format(time()-t0))

        # Test without supervised, and (intentionally) bad priors
        print("\nTesting without supervised, with bad priors (strong)")
        t0 = time()
        gen_model = GenerativeModel(lf_propensity=True)
        gen_model.train(
            L,
            LF_acc_prior_weights=bad_prior_weights,
            reg_type=2,
            reg_param=100 * n,
            candidate_ranges=candidate_ranges
        )
        stats = gen_model.learned_lf_stats()
        accs = stats["Accuracy"]
        coverage = stats["Coverage"]
        print(accs)
        self.assertTrue(np.all(np.abs(accs - np.array(bad_prior)) < tol))
        print("Finished in {0} sec.".format(time()-t0))

    def test_categorical(self):
        LF_acc_priors = [0.75, 0.75, 0.75, 0.75, 0.9]
        print("Generating L...")
        L, labels = self._generate_L(LF_acc_priors)
        print("Running tests for categorical (K=4)...")
        self._test_categorical(L, LF_acc_priors, labels)

    def test_scoped_categorical_small(self):
        LF_acc_priors = [0.75, 0.75, 0.75, 0.75, 0.9]
        print("Generating L...")
        L, labels, candidate_ranges = self._generate_L_scoped_categorical(
            LF_acc_priors)
        print("Running tests for scoped-categorical (K=4)...")
        self._test_categorical(L, LF_acc_priors, labels,
            candidate_ranges=candidate_ranges)

    # def test_scoped_categorical_large(self):
    #     LF_acc_priors = [0.75, 0.75, 0.75, 0.75, 0.9]
    #     print("Generating L...")
    #     L, labels, candidate_ranges = self._generate_L_scoped_categorical(
    #         LF_acc_priors, full_cardinality=100)
    #     print("Running tests for scoped-categorical (K=100)...")
    #     self._test_categorical(L, LF_acc_priors, labels,
    #         candidate_ranges=candidate_ranges)

if __name__ == '__main__':
    unittest.main()

import math
from numbskull.inference import FACTORS
from scipy import sparse
from snorkel.learning.gen_learning import GenerativeModel
import unittest


class TestGenLearning(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_compile_no_deps(self):
        # Defines a label matrix
        L = sparse.lil_matrix((5, 3))

        # The first LF always says yes
        L[0, 0] = 1
        L[1, 0] = 1
        L[2, 0] = 1
        L[3, 0] = 1
        L[4, 0] = 1

        # The second LF votes differently
        L[0, 1] = 1
        L[2, 1] = -1
        L[4, 1] = 1

        # The third LF always abstains

        # Tests compilation
        gen_model = GenerativeModel(lf_prior=False, lf_propensity=False, lf_class_propensity=False)
        gen_model._process_dependency_graph(L, ())
        weight, variable, factor, ftv, domain_mask, n_edges = gen_model._compile(L)

        #
        # Weights
        #
        self.assertEqual(len(weight), 4)

        self.assertFalse(weight[0]['isFixed'])
        self.assertEqual(weight[0]['initialValue'], -1)

        for i in range(1, 4):
            self.assertFalse(weight[i]['isFixed'])
            self.assertTrue(0.9 <= weight[i]['initialValue'] <= 1.1)

        #
        # Variables
        #
        self.assertEqual(len(variable), 20)

        for i in range(5):
            self.assertEqual(variable[i]['isEvidence'], 0)
            self.assertTrue(variable[i]['initialValue'] == 0 or variable[i]['initialValue'] == 1)
            self.assertEqual(variable[i]["dataType"], 1)
            self.assertEqual(variable[i]["cardinality"], 2)

        for i in range(5):
            for j in range(3):
                self.assertEqual(variable[5 + i * 3 + j]['isEvidence'], 1)
                self.assertEqual(variable[5 + i * 3 + j]['initialValue'], L[i, j] + 1)
                self.assertEqual(variable[5 + i * 3 + j]["dataType"], 1)
                self.assertEqual(variable[5 + i * 3 + j]["cardinality"], 3)

        #
        # Factors
        #
        self.assertEqual(len(factor), 20)

        for i in range(5):
            self.assertEqual(factor[i]["factorFunction"], FACTORS["FUNC_DP_GEN_CLASS_PRIOR"])
            self.assertEqual(factor[i]["weightId"], 0)
            self.assertEqual(factor[i]["featureValue"], 1)
            self.assertEqual(factor[i]["arity"], 1)
            self.assertEqual(factor[i]["ftv_offset"], i)

        for i in range(5):
            for j in range(3):
                self.assertEqual(factor[5 + i * 3 + j]["factorFunction"], FACTORS["FUNC_DP_GEN_LF_ACCURACY"])
                self.assertEqual(factor[5 + i * 3 + j]["weightId"], j + 1)
                self.assertEqual(factor[5 + i * 3 + j]["featureValue"], 1)
                self.assertEqual(factor[5 + i * 3 + j]["arity"], 2)
                self.assertEqual(factor[5 + i * 3 + j]["ftv_offset"], 5 + 2 * (i * 3 + j))

        #
        # Factor to Var
        #
        self.assertEqual(len(ftv), 35)

        for i in range(5):
            self.assertEqual(ftv[i]["vid"], i)
            self.assertEqual(ftv[i]["dense_equal_to"], 0)

        for i in range(5):
            for j in range(3):
                self.assertEqual(ftv[5 + 2 * (i * 3 + j)]["vid"], i)
                self.assertEqual(ftv[6 + 2 * (i * 3 + j)]["vid"], 5 + i * 3 + j)
                self.assertEqual(ftv[i]["dense_equal_to"], 0)

        #
        # Domain mask
        #
        self.assertEqual(len(domain_mask), 20)
        for i in range(20):
            self.assertFalse(domain_mask[i])

        # n_edges
        self.assertEqual(n_edges, 35)

if __name__ == '__main__':
    unittest.main()

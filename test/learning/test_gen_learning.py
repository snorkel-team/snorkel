import math
from numbskull.inference import FACTORS
from scipy import sparse
from snorkel.learning.gen_learning import GenerativeModel, DEP_EXCLUSIVE, DEP_REINFORCING, DEP_FIXING, DEP_SIMILAR
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
        gen_model = GenerativeModel(class_prior=True, lf_prior=False, lf_propensity=False, lf_class_propensity=False)
        gen_model._process_dependency_graph(L, ())
        weight, variable, factor, ftv, domain_mask, n_edges = gen_model._compile(L, None, 1.0)

        #
        # Weights
        #
        self.assertEqual(len(weight), 4)

        self.assertFalse(weight[0]['isFixed'])
        self.assertEqual(weight[0]['initialValue'], 0.0)

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
            self.assertEqual(variable[i]["dataType"], 0)
            self.assertEqual(variable[i]["cardinality"], 2)

        for i in range(5):
            for j in range(3):
                self.assertEqual(variable[5 + i * 3 + j]['isEvidence'], 1)
                self.assertEqual(variable[5 + i * 3 + j]['initialValue'], L[i, j] + 1)
                self.assertEqual(variable[5 + i * 3 + j]["dataType"], 0)
                self.assertEqual(variable[5 + i * 3 + j]["cardinality"], 3)

        #
        # Factors
        #
        self.assertEqual(len(factor), 20)

        for i in range(5):
            self.assertEqual(factor[i]["factorFunction"], FACTORS["DP_GEN_CLASS_PRIOR"])
            self.assertEqual(factor[i]["weightId"], 0)
            self.assertEqual(factor[i]["featureValue"], 1)
            self.assertEqual(factor[i]["arity"], 1)
            self.assertEqual(factor[i]["ftv_offset"], i)

        for i in range(5):
            for j in range(3):
                self.assertEqual(factor[5 + i * 3 + j]["factorFunction"], FACTORS["DP_GEN_LF_ACCURACY"])
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
                self.assertEqual(ftv[5 + 2 * (i * 3 + j)]["dense_equal_to"], 0)
                self.assertEqual(ftv[6 + 2 * (i * 3 + j)]["dense_equal_to"], 0)

        #
        # Domain mask
        #
        self.assertEqual(len(domain_mask), 20)
        for i in range(20):
            self.assertFalse(domain_mask[i])

        # n_edges
        self.assertEqual(n_edges, 35)

    def test_compile_with_deps(self):
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

        # Defined dependencies
        deps = []
        deps.append((0, 1, DEP_SIMILAR))
        deps.append((0, 2, DEP_SIMILAR))

        deps.append((0, 1, DEP_FIXING))
        deps.append((0, 2, DEP_REINFORCING))
        deps.append((1, 2, DEP_EXCLUSIVE))

        # Tests compilation
        gen_model = GenerativeModel(class_prior=False, lf_prior=False, lf_propensity=True, lf_class_propensity=False)
        gen_model._process_dependency_graph(L, deps)
        weight, variable, factor, ftv, domain_mask, n_edges = gen_model._compile(L, None, 1.0)

        #
        # Weights
        #
        self.assertEqual(len(weight), 11)

        for i in range(3):
            self.assertFalse(weight[i]['isFixed'])
            self.assertTrue(0.9 <= weight[i]['initialValue'] <= 1.1)
        for i in range(3, 11):
            self.assertFalse(weight[i]['isFixed'])
            self.assertEqual(weight[i]['initialValue'], 0.0)

        #
        # Variables
        #
        self.assertEqual(len(variable), 20)

        for i in range(5):
            self.assertEqual(variable[i]['isEvidence'], 0)
            self.assertTrue(variable[i]['initialValue'] == 0 or variable[i]['initialValue'] == 1)
            self.assertEqual(variable[i]["dataType"], 0)
            self.assertEqual(variable[i]["cardinality"], 2)

        for i in range(5):
            for j in range(3):
                self.assertEqual(variable[5 + i * 3 + j]['isEvidence'], 1)
                self.assertEqual(variable[5 + i * 3 + j]['initialValue'], L[i, j] + 1)
                self.assertEqual(variable[5 + i * 3 + j]["dataType"], 0)
                self.assertEqual(variable[5 + i * 3 + j]["cardinality"], 3)

        #
        # Factors
        #
        self.assertEqual(len(factor), 55)

        f_offset = 0
        ftv_offset = 0
        for i in range(5):
            for j in range(3):
                self.assertEqual(factor[f_offset + i * 3 + j]["factorFunction"], FACTORS["DP_GEN_LF_ACCURACY"])
                self.assertEqual(factor[f_offset + i * 3 + j]["weightId"], j)
                self.assertEqual(factor[f_offset + i * 3 + j]["featureValue"], 1)
                self.assertEqual(factor[f_offset + i * 3 + j]["arity"], 2)
                self.assertEqual(factor[f_offset + i * 3 + j]["ftv_offset"], ftv_offset + 2 * (i * 3 + j))

        f_offset = 15
        ftv_offset = 30
        for i in range(5):
            for j in range(3):
                self.assertEqual(factor[f_offset + i * 3 + j]["factorFunction"], FACTORS["DP_GEN_LF_PROPENSITY"])
                self.assertEqual(factor[f_offset + i * 3 + j]["weightId"], 3 + j)
                self.assertEqual(factor[f_offset + i * 3 + j]["featureValue"], 1)
                self.assertEqual(factor[f_offset + i * 3 + j]["arity"], 1)
                self.assertEqual(factor[f_offset + i * 3 + j]["ftv_offset"], ftv_offset + (i * 3 + j))

        f_offset = 30
        ftv_offset = 45
        for i in range(5):
            self.assertEqual(factor[f_offset + i]["factorFunction"], FACTORS["EQUAL"])
            self.assertEqual(factor[f_offset + i]["weightId"], 6)
            self.assertEqual(factor[f_offset + i]["featureValue"], 1)
            self.assertEqual(factor[f_offset + i]["arity"], 2)
            self.assertEqual(factor[f_offset + i]["ftv_offset"], ftv_offset + 2 * i)

        f_offset = 35
        ftv_offset = 55
        for i in range(5):
            self.assertEqual(factor[f_offset + i]["factorFunction"], FACTORS["EQUAL"])
            self.assertEqual(factor[f_offset + i]["weightId"], 7)
            self.assertEqual(factor[f_offset + i]["featureValue"], 1)
            self.assertEqual(factor[f_offset + i]["arity"], 2)
            self.assertEqual(factor[f_offset + i]["ftv_offset"], ftv_offset + 2 * i)

        f_offset = 40
        ftv_offset = 65
        for i in range(5):
            self.assertEqual(factor[f_offset + i]["factorFunction"], FACTORS["DP_GEN_DEP_FIXING"])
            self.assertEqual(factor[f_offset + i]["weightId"], 8)
            self.assertEqual(factor[f_offset + i]["featureValue"], 1)
            self.assertEqual(factor[f_offset + i]["arity"], 3)
            self.assertEqual(factor[f_offset + i]["ftv_offset"], ftv_offset + 3 * i)

        f_offset = 45
        ftv_offset = 80
        for i in range(5):
            self.assertEqual(factor[f_offset + i]["factorFunction"], FACTORS["DP_GEN_DEP_REINFORCING"])
            self.assertEqual(factor[f_offset + i]["weightId"], 9)
            self.assertEqual(factor[f_offset + i]["featureValue"], 1)
            self.assertEqual(factor[f_offset + i]["arity"], 3)
            self.assertEqual(factor[f_offset + i]["ftv_offset"], ftv_offset + 3 * i)

        f_offset = 50
        ftv_offset = 95
        for i in range(5):
            self.assertEqual(factor[f_offset + i]["factorFunction"], FACTORS["DP_GEN_DEP_EXCLUSIVE"])
            self.assertEqual(factor[f_offset + i]["weightId"], 10)
            self.assertEqual(factor[f_offset + i]["featureValue"], 1)
            self.assertEqual(factor[f_offset + i]["arity"], 2)
            self.assertEqual(factor[f_offset + i]["ftv_offset"], ftv_offset + 2 * i)

        #
        # Factor to Var
        #
        self.assertEqual(len(ftv), 105)

        ftv_offset = 0
        for i in range(5):
            for j in range(3):
                self.assertEqual(ftv[ftv_offset + 2 * (i * 3 + j)]["vid"], i)
                self.assertEqual(ftv[ftv_offset + 2 * (i * 3 + j)]["dense_equal_to"], 0)
                self.assertEqual(ftv[ftv_offset + 2 * (i * 3 + j) + 1]["vid"], 5 + i * 3 + j)
                self.assertEqual(ftv[ftv_offset + 2 * (i * 3 + j) + 1]["dense_equal_to"], 0)

        ftv_offset = 30
        for i in range(5):
            for j in range(3):
                self.assertEqual(ftv[ftv_offset + (i * 3 + j)]["vid"], 5 + i * 3 + j)
                self.assertEqual(ftv[ftv_offset + (i * 3 + j)]["dense_equal_to"], 0)

        ftv_offset = 45
        for i in range(5):
            self.assertEqual(ftv[ftv_offset + 2 * i]["vid"], 5 + i * 3)
            self.assertEqual(ftv[ftv_offset + 2 * i]["dense_equal_to"], 0)

            self.assertEqual(ftv[ftv_offset + 2 * i + 1]["vid"], 5 + i * 3 + 1)
            self.assertEqual(ftv[ftv_offset + 2 * i + 1]["dense_equal_to"], 0)

        ftv_offset = 55
        for i in range(5):
            self.assertEqual(ftv[ftv_offset + 2 * i]["vid"], 5 + i * 3)
            self.assertEqual(ftv[ftv_offset + 2 * i]["dense_equal_to"], 0)

            self.assertEqual(ftv[ftv_offset + 2 * i + 1]["vid"], 5 + i * 3 + 2)
            self.assertEqual(ftv[ftv_offset + 2 * i + 1]["dense_equal_to"], 0)

        ftv_offset = 65
        for i in range(5):
            self.assertEqual(ftv[ftv_offset + 3 * i]["vid"], i)
            self.assertEqual(ftv[ftv_offset + 3 * i]["dense_equal_to"], 0)

            self.assertEqual(ftv[ftv_offset + 3 * i + 1]["vid"], 5 + i * 3)
            self.assertEqual(ftv[ftv_offset + 3 * i + 1]["dense_equal_to"], 0)

            self.assertEqual(ftv[ftv_offset + 3 * i + 2]["vid"], 5 + i * 3 + 1)
            self.assertEqual(ftv[ftv_offset + 3 * i + 2]["dense_equal_to"], 0)

        ftv_offset = 80
        for i in range(5):
            self.assertEqual(ftv[ftv_offset + 3 * i]["vid"], i)
            self.assertEqual(ftv[ftv_offset + 3 * i]["dense_equal_to"], 0)

            self.assertEqual(ftv[ftv_offset + 3 * i + 1]["vid"], 5 + i * 3)
            self.assertEqual(ftv[ftv_offset + 3 * i + 1]["dense_equal_to"], 0)

            self.assertEqual(ftv[ftv_offset + 3 * i + 2]["vid"], 5 + i * 3 + 2)
            self.assertEqual(ftv[ftv_offset + 3 * i + 2]["dense_equal_to"], 0)

        ftv_offset = 95
        for i in range(5):
            self.assertEqual(ftv[ftv_offset + 2 * i]["vid"], 5 + i * 3 + 1)
            self.assertEqual(ftv[ftv_offset + 2 * i]["dense_equal_to"], 0)

            self.assertEqual(ftv[ftv_offset + 2 * i + 1]["vid"], 5 + i * 3 + 2)
            self.assertEqual(ftv[ftv_offset + 2 * i + 1]["dense_equal_to"], 0)

        #
        # Domain mask
        #
        self.assertEqual(len(domain_mask), 20)
        for i in range(20):
            self.assertFalse(domain_mask[i])

        # n_edges
        self.assertEqual(n_edges, 105)

if __name__ == '__main__':
    unittest.main()

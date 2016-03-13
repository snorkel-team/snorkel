import unittest, os
import numpy as np
import scipy.sparse as sparse

from ddlite import *

class TestInference(unittest.TestCase):
    
    def setUp(self):
        self.n = None
        self.F = None
        self.R = None
        self.w_opt = None
        self.gt = None
        self.X = None
    
    def tearDown(self):
        pass
    
    def test_ridge_logistic_regression_easy(self):
        print "Running easy logistic regression test"
        # Setup problem
        self.n = 50
        n_feats, n_goodfeats = 10, 8
        n_rules, n_goodrules = 4, 4
        self._make_feats(n_feats, n_goodfeats)
        self._make_rules(n_rules, n_goodrules, n_feats, n_goodfeats, 0, 
                         n_goodfeats * 1/2)
        self._make_X()
        # Learn params
        w0 = np.concatenate([np.ones(n_rules), np.zeros(n_feats)])
        w = learn_params(self.X, 100, sample=False, verbose=True, w0=w0,
                         mu=0.01)
        self.assertGreater(np.mean(np.sign(self.X.dot(w)) == self.gt),
                           0.9)
    
    def test_ridge_logistic_regression_medium(self):
        print "Running medium logistic regression test"
        # Setup problem
        self.n = 250
        n_feats, n_goodfeats = 25, 20
        n_rules, n_goodrules = 4, 3
        self._make_feats(n_feats, n_goodfeats)
        self._make_rules(n_rules, n_goodrules, n_feats, n_goodfeats, 0, 
                         n_goodfeats * 1/2)
        self._make_X()
        # Learn params
        w0 = np.concatenate([np.ones(n_rules), np.zeros(n_feats)])
        w = learn_params(self.X, 500, sample=False, verbose=True, w0=w0, 
                         mu=0.01)
        self.assertGreater(np.mean(np.sign(self.X.dot(w)) == self.gt), 
                           0.9)
        
    def test_ridge_logistic_regression_hard(self):
        print "Running hard logistic regression test"
        # Setup problem
        self.n = 250
        n_feats, n_goodfeats = 25, 20
        n_rules, n_goodrules = 4, 3
        self._make_feats(n_feats, n_goodfeats)
        self._make_rules(n_rules, n_goodrules, n_feats, n_goodfeats, 0.05,
                         n_goodfeats * 1/2)
        self._make_X()
        # Learn params
        w0 = np.concatenate([np.ones(n_rules), np.zeros(n_feats)])
        w = learn_params(self.X, 500, sample=False, verbose=True, w0=w0,
                         mu=0.01)
        self.assertGreater(np.mean(np.sign(self.X.dot(w)) == self.gt), 
                           0.8)
        
    def test_logistic_regression_sparse(self):
        print "Running logistic regression test with sparse operations"
        # Setup problem
        self.n = 100
        n_feats, n_goodfeats = 2500, 2000
        n_rules, n_goodrules = 40, 30
        self._make_feats(n_feats, n_goodfeats)
        self._make_rules(n_rules, n_goodrules, n_feats, n_goodfeats, 0.05,
                         n_goodfeats * 1/2)
        self._make_X()
        # Learn params
        w0 = np.concatenate([np.ones(n_rules), np.zeros(n_feats)])
        w_d = learn_params(self.X, 500, sample=False, verbose=True, w0=w0,
                           mu=0.01)
        w_s = learn_params(sparse.csr_matrix(self.X), 500, sample=False,
                           verbose=True, w0=w0, mu=0.01)
        # Check sparse solution is close to dense solution
        self.assertLessEqual(np.linalg.norm(w_s - w_d), 1e-4)
    
    def test_logistic_regression_sample(self):
        print "Running logistic regression test with sampling"
        # Setup problem
        self.n = 1000
        n_feats, n_goodfeats = 15000, 10000
        n_rules, n_goodrules = 400, 50
        self._make_feats(n_feats, n_goodfeats, c=0.005, sp=True)
        self._make_rules(n_rules, n_goodrules, n_feats, n_goodfeats, 0.05,
                         n_goodfeats * 0.005)
        self._make_X()       
        X_sparse = sparse.csr_matrix(self.X)        
        # Learn params
        w0 = np.concatenate([np.ones(n_rules), np.zeros(n_feats)])
        w = learn_params(X_sparse, 500, sample=False,
                         verbose=True, w0=w0, mu=0.01)
        w_samp = learn_params(X_sparse, 500, sample=True, nSamples=100,
                              verbose=True, w0=w0, mu=0.01)
        # Check sparse solution is close to dense solution
        self.assertLessEqual(np.linalg.norm(w_samp - w), 0.01)
        
    def _make_feats(self, nf, ngf, c=0.5, sp=False):
        # Make NF binary features
        self.F = (np.random.rand(self.n, nf) < c).astype(float)
        if sp:
            self.F = sparse.csr_matrix(self.F)
        # Set the ground truth by summing first NGF features
        # If at least 1/2 are 1's, then it's a positive case
        cutoff = ngf * c
        self.w_opt = np.concatenate([np.ones(ngf), np.zeros(nf - ngf)])
        self.gt = 2*((self.F.dot(self.w_opt) >= cutoff).astype(float)) - 1
    
    def _make_rules(self, nr, ngr, nf, ngf, strength, cutoff):
        # Make NGR good rules
        Rg = np.vstack([self.F.dot(np.concatenate([(np.random.rand(ngf) >
                                                    strength).astype(float), 
                                                  np.zeros(nf-ngf)]))
                        for _ in xrange(ngr)])
        # Make NR - NGR random rules
        if nr != ngr:
            Rb = np.vstack([ngf * np.random.rand(self.n) 
                           for _ in xrange(nr - ngr)])
            R = np.vstack([Rg, Rb])
        else:
            R = Rg
        # Convert to rule output (no zeros)
        self.R = (2*((R >= cutoff).astype(float)) - 1).T
    
    def _make_X(self):
        if sparse.issparse(self.F):
            self.X = sparse.hstack([self.R, self.F], format='csr')
        else:
            self.X = np.hstack([self.R, self.F])


if __name__ == '__main__':
    os.chdir('../')
    unittest.main()
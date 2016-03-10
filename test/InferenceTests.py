import unittest, os
import numpy as np

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
        self._make_rules(n_rules, n_goodrules, n_feats, n_goodfeats, 0, n_goodfeats * 1/2)
        self._make_X()
        # Learn params
        w0 = np.concatenate([np.ones(n_rules), np.zeros(n_feats)])
        w = learn_params(self.X, 100, sample=False, verbose=True, w0=w0, mu=0.01)
        self.assertGreater(np.mean(np.sign(np.dot(self.X.T, w)) == self.gt), 0.9)
    
    def test_ridge_logistic_regression_medium(self):
        print "Running medium logistic regression test"
        # Setup problem
        self.n = 250
        n_feats, n_goodfeats = 25, 20
        n_rules, n_goodrules = 4, 3
        self._make_feats(n_feats, n_goodfeats)
        self._make_rules(n_rules, n_goodrules, n_feats, n_goodfeats, 0, n_goodfeats * 1/2)
        self._make_X()
        # Learn params
        w0 = np.concatenate([np.ones(n_rules), np.zeros(n_feats)])
        w = learn_params(self.X, 500, sample=False, verbose=True, w0=w0, mu=0.01)
        self.assertGreater(np.mean(np.sign(np.dot(self.X.T, w)) == self.gt), 0.9)
        
    def test_ridge_logistic_regression_hard(self):
        print "Running hard logistic regression test"
        # Setup problem
        self.n = 250
        n_feats, n_goodfeats = 25, 20
        n_rules, n_goodrules = 4, 3
        self._make_feats(n_feats, n_goodfeats)
        self._make_rules(n_rules, n_goodrules, n_feats, n_goodfeats, 0.05, n_goodfeats * 1/2)
        self._make_X()
        # Learn params
        w0 = np.concatenate([np.ones(n_rules), np.zeros(n_feats)])
        w = learn_params(self.X, 500, sample=False, verbose=True, w0=w0, mu=0.01)
        self.assertGreater(np.mean(np.sign(np.dot(self.X.T, w)) == self.gt), 0.8)
    
    def _make_feats(self, nf, ngf):
        # Make NF binary features
        self.F = (np.random.rand(self.n, nf) > 0.5).astype(float)
        # Set the ground truth by summing first NGF features
        # If at least 1/2 are 1's, then it's a positive case
        cutoff = ngf * 1/2
        self.w_opt = np.concatenate([np.ones(ngf), np.zeros(nf - ngf)])
        self.gt = 2*((self.F.dot(self.w_opt) >= cutoff).astype(float)) - 1
    
    def _make_rules(self, nr, ngr, nf, ngf, strength, cutoff):
        # Make NGR good rules
        Rg = np.vstack([self.F.dot(np.concatenate([(np.random.rand(ngf) > strength).astype(float), 
                                                   np.zeros(nf-ngf)]))
                        for _ in xrange(ngr)])
        # Make NR - NGR random rules
        if nr != ngr:
            Rb = np.vstack([ngf * np.random.rand(self.n) for _ in xrange(nr - ngr)])
            R = np.vstack([Rg, Rb])
        else:
            R = Rg
        # Convert to rule output (no zeros)
        self.R = (2*((R >= cutoff).astype(float)) - 1).T
    
    def _make_X(self):
        self.X = np.vstack([self.R.T, self.F.T])


if __name__ == '__main__':
    os.chdir('../')
    unittest.main()
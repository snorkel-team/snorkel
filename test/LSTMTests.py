import os, sys, unittest, cPickle

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from ddlite import *

class TestLSTM(unittest.TestCase):

  def test_accuracy(self):
    np.random.seed(seed=1701)

    E = Entities('data/lstm_test/gene_tag_saved_entities_v6.pkl')

    feats = None
    pkl_f = 'data/lstm_test/gene_tag_feats_v1.pkl'
    with open(pkl_f, 'rb') as f:
      feats = cPickle.load(f)

    DDL = DDLiteModel(E, feats)
    print "Extracted {} features for each of {} mentions".format(DDL.num_feats(), DDL.num_candidates())

    with open('data/lstm_test/gt/uids.pkl', 'rb') as f:
      uids = cPickle.load(f)
    with open('data/lstm_test/gt/gt.pkl', 'rb') as f:
      gt = cPickle.load(f)
    
    DDL.update_gt(gt[:50], uids=uids[:50])
    DDL.set_holdout(validation_frac=0.5)
    DDL.update_gt(gt[50:], uids=uids[50:])

    def LF_gene(m):
      return 1 if ('gene' in m.post_window('lemmas')) or ('gene' in m.pre_window('lemmas')) else 0
    def LF_gene_dp(m):
      return 1 if 'gene' in [m.lemmas[m.dep_parents[i] - 1] for i in m.idxs] else 0
    def LF_genotype_dp(m):
      return 1 if 'genotype' in [m.lemmas[m.dep_parents[i] - 1] for i in m.idxs] else 0
    def LF_mutant(m):
      return 1 if ('mutant' in m.post_window('lemmas')) or ('mutant' in m.pre_window('lemmas')) else 0
    def LF_variant(m):
      return 1 if ('variant' in m.post_window('lemmas')) or ('variant' in m.pre_window('lemmas')) else 0
    def LF_express(m):
      return 1 if ('express' in m.post_window('lemmas')) or ('express' in m.pre_window('lemmas')) else 0
    def LF_mutation(m):
      return 1 if 'mutation' in [m.lemmas[m.dep_parents[i] - 1] for i in m.idxs] else 0
    def LF_JJ(m):
      return 1 if 'JJ' in m.post_window('poses') else 0
    def LF_IN(m):
      return 1 if 'IN' in m.pre_window('poses', 1) else 0

    def LF_dna(m):
      return -1 if 'DNA' in m.mention('words') else 0
    def LF_rna(m):
      return -1 if 'RNA' in m.mention('words') else 0
    def LF_snp(m):
      return -1 if 'SNP' in m.mention('words') else 0
    def LF_protein(m):
      return -1 if 'protein' in m.pre_window('lemmas') else 0
    def LF_LRB(m):
      return -1 if '-LRB-' in m.post_window('poses', 1) else 0
    def LF_RRB(m):
      return -1 if '-RRB-' in m.post_window('poses', 1) else 0
    def LF_dev_dp(m):
      return -1 if 'development' in [m.lemmas[m.dep_parents[i] - 1] for i in m.idxs] else 0
    def LF_protein_dp(m):
      return -1 if 'protein' in [m.lemmas[m.dep_parents[i] - 1] for i in m.idxs] else 0
    def LF_network_dp(m):
      return -1 if 'network' in [m.lemmas[m.dep_parents[i] - 1] for i in m.idxs] else 0
    def LF_JJ_dp(m):
      return -1 if 'JJ' in [m.poses[m.dep_parents[i] - 1] for i in m.idxs] else 0
    def LF_NNP(m):
      return -1 if 'NNP' in m.mention('poses') else 0

    LFs = [LF_JJ, LF_JJ_dp, LF_NNP, LF_RRB, LF_dev_dp, LF_dna, LF_express, LF_gene, LF_gene_dp,
           LF_genotype_dp, LF_mutant, LF_mutation, LF_network_dp, LF_protein, LF_protein_dp,
           LF_rna, LF_snp, LF_variant, LF_IN, LF_LRB]
    DDL.apply_lfs(LFs, clear=False)

    mu_seq = np.ravel([1e-9, 1e-5, 1e-3, 1e-1])
    DDL.learn_weights(sample=False, n_iter_lf=500, n_iter_feats=3000, mu_lf=1e-7, mu_seq_feats=mu_seq, verbose=True, w0_mult_lf=1, rate=0.01, alpha=0.5, bias=False)
    idxs, self.gt = DDL.get_labeled_ground_truth(subset=DDL.holdout())

    DDL.lstm_learn_weights(n_iter=300,verbose=True, contain_mention=True, word_window_length=0, ignore_case=False)
    self.lstm_pred=np.array(DDL.lstm_pred)[DDL.holdout()]

    self.assertGreaterEqual(np.mean(self.lstm_pred==self.gt), 0.7)

if __name__ == '__main__':
    unittest.main()
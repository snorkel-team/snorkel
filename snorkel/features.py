import os, sys
from collections import defaultdict
import scipy.sparse as sparse
from .models import Candidate, CandidateSet, Feature
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree
from utils import get_as_dict
from entity_features import *


class SessionFeaturizer(object):
    def __init__(self):
        self.feat_index = None
        self.inv_index  = None

    def get_feats(self, candidate):
        raise NotImplementedError()

    def featurize_and_save(self, session, candidates):
        """Featurize a collection of Candidate objects, using *get_feat*, then add to session to persist to DB"""
        feature_set = set()
        for candidate in candidates:
            feature_set.clear()
            for feat_name in self.get_feats(candidate):
                f = Feature(candidate=candidate, name=feat_name)
                if f not in feature_set:
                    session.add(f)
                    feature_set.add(f)
        session.commit()

    def load_feats(self, session, cand_set, yield_per=1000):
        """Load features for this candidate set _sorting by candidate id_"""
        for f in session.query(Feature.name, Feature.candidate_id).join(Candidate) \
                        .filter(Candidate.set == cand_set).order_by(Candidate.id).yield_per(yield_per):
            yield f

    def get_cid_map(self, session, cand_set):
        """Get a map from each candidate id to a row number, default according to sorted order"""
        cids = session.query(Candidate.id).filter(Candidate.set == cand_set).order_by(Candidate.id)
        return dict((c[0], i) for i,c in enumerate(cids))
    
    def load_and_fit(self, session, candidate_set):
        """
        Loads the features for a given Candidate Set and represents as a CSR sparse matrix F,
        where F[i,j] \in {0,1} indicates whether the ith candidate has the jth feature,
        also fitting the feature space to this candidate set, setting *self.feat_index* and *self.inv_index*.
        """
        # First, we need a mapping from cid -> row number
        c_index = self.get_cid_map(session, candidate_set)

        # Next, we get the feature -> row number mappings
        f_index = defaultdict(list)
        for f, cid in self.load_feats(session, candidate_set):
            f_index[f].append(c_index[cid])

        # Assemble and return sparse feature matrix
        # Also assemble reverse index of feature matrix index -> feature verbose name
        self.feat_index = {}
        self.inv_index  = {}
        F               = sparse.lil_matrix((len(candidate_set), len(f_index.keys())))
        for j,f in enumerate(f_index.keys()):
            self.feat_index[f] = j
            self.inv_index[j]  = f
            for i in f_index[f]:
                F[i,j] = 1
        return F.tocsr()

    def load(self, session, candidate_set):
        """
        Loads the features for a given Candidate Set and represents as a CSR sparse matrix F,
        where F[i,j] \in {0,1} indicates whether the ith candidate has the jth feature.
        """
        # First, we need a mapping from cid -> row number
        c_index = self.get_cid_map(session, candidate_set)

        # Next, fit the candidate set to the existing feature space in self.feat_index
        F = sparse.lil_matrix((len(candidate_set), len(self.feat_index.keys())))
        for f in self.load_feats(session, candidate_set):
            if self.feat_index.has_key(f.name):
                F[c_index[f.candidate_id], self.feat_index[f.name]] = 1
        return F.tocsr()


class SpanFeaturizer(SessionFeaturizer):
    """Featurizer for Span objects"""
    def __init__(self):
        self.feature_generator = compile_entity_feature_generator()
        super(SpanFeaturizer, self).__init__()

    def get_feats(self, span):
        sent    = get_as_dict(span.context)
        xmltree = corenlp_to_xmltree(sent)
        sidxs   = range(span.get_word_start(), span.get_word_end() + 1)
        if len(sidxs) > 0:

            # Add DDLIB entity features
            for f in get_ddlib_feats(sent, sidxs):
                yield 'DDL_' + f

            # Add TreeDLib entity features
            for f in self.feature_generator(xmltree.root, sidxs):
                yield 'TDL_' + f


class SpanPairFeaturizer(SessionFeaturizer):
    """Featurizer for SpanPair objects"""
    def __init__(self):
        self.feature_generator = compile_relation_feature_generator()
        super(SpanPairFeaturizer, self).__init__()

    def get_feats(self, span_pair):
        xmltree = corenlp_to_xmltree(get_as_dict(span_pair.span0.context))
        s1_idxs = range(span_pair.span0.get_word_start(), span_pair.span0.get_word_end() + 1)
        s2_idxs = range(span_pair.span1.get_word_start(), span_pair.span1.get_word_end() + 1)
        if len(s1_idxs) > 0 and len(s2_idxs) > 0:

            # Apply TDL features
            for f in self.feature_generator(xmltree.root, s1_idxs, s2_idxs):
                yield 'TDL_' + f

import os, sys
from collections import defaultdict
import scipy.sparse as sparse
from snorkel import SnorkelSession
from snorkel.features import SpanPairFeaturizer
from snorkel.models import CandidateSet, Feature
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree, XMLTree
from utils import get_as_dict
from entity_features import *


def main():
    session = SnorkelSession()

    # Loads candidates
    train_candidates = session.query(CandidateSet).filter(CandidateSet.name == 'Train Candidates').one()
    test_candidates = session.query(CandidateSet).filter(CandidateSet.name == 'Test Candidates').one()

    # Generates features
    featurizer = SpanPairFeaturizer()
    feature_set = set()
    for candidates in [train_candidates, test_candidates]:
        for candidate in candidates:
            feature_set.clear()
            for feature, i in featurizer.get_feats((candidate,)):
                f = Feature(candidate=candidate, name=feature)
                if f not in feature_set:
                    session.add(f)
                    feature_set.add(f)

    session.commit()


def load_all_features(candidate_set, session):
    """
    Given a CandidateSet and Session, generates (Candidate, <feature name>) pairs for all Candidates in the set.
    """
    for x in session.query(Candidate).filter(Candidate.set == candidate_set).join(Feature):
        yield x


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
    
    def load_and_fit(self, session, candidate_set):
        """
        Loads the features for a given Candidate Set and represents as a sparse matrix F,
        where F[i,j] \in {0,1} indicates whether the ith candidate has the jth feature.
        This method also refreshes *self.feat_index* and *self.inv_index*
        """
        f_index = defaultdict(list)
        for f, i in self.get_feats(candidates):
            f_index[f].append(i)

        # Assemble and return sparse feature matrix
        # Also assemble reverse index of feature matrix index -> feature verbose name
        self.feat_index = {}
        self.inv_index  = {}
        F               = sparse.lil_matrix((len(candidates), len(f_index.keys())))
        for j,f in enumerate(f_index.keys()):
            self.feat_index[f] = j
            self.inv_index[j]  = f
            for i in f_index[f]:
                F[i,j] = 1
        return F


class SpanFeaturizer(SessionFeaturizer):
    """Featurizer for Span objects"""
    def __init__(self):
        self.feature_generator = compile_entity_feature_generator()
        super(SpanFeaturizer, self).__init__()

    def get_feats(self, span):
        sent    = get_as_dict(span.context)
        xmltree = corenlp_to_xmltree(sent)
        sidxs   = range(span.get_word_start(), span.get_word_end() + 1)

        # Add DDLIB entity features
        for f in get_ddlib_feats(sent, sidxs):
            yield 'DDL_' + f

        # Add TreeDLib entity features
        for f in self.feature_generator(xmltree.root, sidxs):
            yield 'TDL_' + f


class Featurizer(object):
    def __init__(self):
        self.feat_index = None
        self.inv_index  = None

    def get_feats(self, candidates):
        raise NotImplementedError()

    def fit_transform(self, candidates):
        """Assembles the set of features to be used, and applies this transformation to the candidates"""
        # Assemble and return the sparse feature matrix
        f_index = defaultdict(list)
        for f, i in self.get_feats(candidates):
            f_index[f].append(i)

        # Assemble and return sparse feature matrix
        # Also assemble reverse index of feature matrix index -> feature verbose name
        self.feat_index = {}
        self.inv_index  = {}
        F               = sparse.lil_matrix((len(candidates), len(f_index.keys())))
        for j,f in enumerate(f_index.keys()):
            self.feat_index[f] = j
            self.inv_index[j]  = f
            for i in f_index[f]:
                F[i,j] = 1
        return F

    def transform(self, candidates):
        """Given feature set has already been fit, simply apply to candidates."""
        F = sparse.lil_matrix((len(candidates), len(self.feat_index.keys())))
        for f,i in self.get_feats(candidates):
            if self.feat_index.has_key(f):
                F[i,self.feat_index[f]] = 1
        return F


class SpanPairFeaturizer(Featurizer):
    """Featurizer for SpanPair objects"""

    def __init__(self):
        self.feature_generator = compile_relation_feature_generator()
        super(SpanPairFeaturizer, self).__init__()

    def get_feats(self, span_pairs):
        for i,sp in enumerate(span_pairs):
            xmltree = corenlp_to_xmltree(get_as_dict(sp.span0.context))
            s1_idxs = range(sp.span0.get_word_start(), sp.span0.get_word_end() + 1)
            s2_idxs = range(sp.span1.get_word_start(), sp.span1.get_word_end() + 1)

            # Apply TDL features
            for f in self.feature_generator(xmltree.root, s1_idxs, s2_idxs):
                yield 'TDL_' + f, i



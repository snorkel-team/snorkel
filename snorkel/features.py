import os, sys
import numpy as np
from pandas import DataFrame
from collections import defaultdict
import scipy.sparse as sparse
from .models import Candidate, CandidateSet, Feature, TemporarySpan, Phrase
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree
from utils import get_as_dict
from entity_features import *


def get_span_feats(candidate):
    args = candidate.get_arguments()
    if not (isinstance(args[0], TemporarySpan)):
        raise ValueError("Accepts Span-type arguments, %s-type found." % type(candidate))

    # Unary candidates
    if len(args) == 1:
        get_tdl_feats = compile_entity_feature_generator()
        get_tabledlib_feats =  tabledlib_unary_features
        span          = args[0]
        sent          = get_as_dict(span.parent)
        xmltree       = corenlp_to_xmltree(sent)
        sidxs         = range(span.get_word_start(), span.get_word_end() + 1)
        if len(sidxs) > 0:

            # Add DDLIB entity features
            for f in get_ddlib_feats(sent, sidxs):
                yield 'DDL_' + f, 1

            # Add TreeDLib entity features
            for f in get_tdl_feats(xmltree.root, sidxs):
                yield 'TDL_' + f, 1

            # Add TableDLib entity features (if applicable)
            if isinstance(span.parent, Phrase):
                for f in get_tabledlib_feats(span):
                    yield 'TAB_' + f, 1

    # Binary candidates
    elif len(args) == 2:
        get_tdl_feats = compile_relation_feature_generator()
        get_tabledlib_feats = tabledlib_binary_features
        span1, span2  = args
        xmltree       = corenlp_to_xmltree(get_as_dict(span1.parent))
        s1_idxs       = range(span1.get_word_start(), span1.get_word_end() + 1)
        s2_idxs       = range(span2.get_word_start(), span2.get_word_end() + 1)
        if len(s1_idxs) > 0 and len(s2_idxs) > 0:

            # Apply TreeDLib relation features
            for f in get_tdl_feats(xmltree.root, s1_idxs, s2_idxs):
                yield 'TDL_' + f, 1
            
            # Add TableDLib relation features (if applicable)
            if isinstance(span1.parent, Phrase) or isinstance(span2.parent, Phrase):
                for f in get_tabledlib_feats(span1, span2, s1_idxs, s2_idxs):
                    yield 'TAB_' + f, 1
    else:
        raise NotImplementedError("Only handles unary and binary candidates currently")


# TODO: DELETE!
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

        # Return F in CSR sparse format
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
    
    def top_features(self, w, n_max=100):
        """Return a DataFrame of highest (abs)-weighted features"""
        idxs = np.argsort(np.abs(w))[::-1][:n_max]
        d = {'j': idxs, 'w': [w[i] for i in idxs]}
        return DataFrame(data=d, index=[self.inv_index[i] for i in idxs])


# TODO: DELETE!
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


# TODO: DELETE!
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

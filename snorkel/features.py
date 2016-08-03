import os, sys
from collections import defaultdict
import numpy as np
import scipy.sparse as sparse
import itertools

# Feature modules
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree, XMLTree
from utils import get_as_dict
from entity_features import *


class Featurizer(object):
    """
    A Featurizer applies a set of **feature generators** to each Candidate,
    based on (i) the arity of the candidate, and (ii) the _associated Contexts_.

    The transform() function takes in N candidates, and returns an N x F sparse matrix,
    where F is the dimension of the feature space.
    """
    def __init__(self, arity=1):
        self.arity          = arity
        self.feat_index     = None
        self.feat_inv_index = None

    def _generate_context_feats(self, get_feats, prefix, candidates):
        """
        Given a function that given a candidate, generates features, _using a specific context_,
        and a unique prefix string for this context, return a generator over features (as strings).
        """
        for i,c in enumerate(candidates):
            for f in get_feats(c):
                yield i, prefix + f

    # TODO: Take this out...
    def _preprocess_candidates(self, candidates):
        return candidates

    def _match_contexts(self, candidates):
        """Given the candidates, and using _generate_context_feats, return a list of generators."""
        raise NotImplementedError()

    def transform(self, candidates):
        """Given feature set has already been fit, simply apply to candidates."""
        F                  = sparse.lil_matrix((len(candidates), len(self.feat_index.keys())))
        feature_generators = self._match_contexts(self._preprocess_candidates(candidates))
        for i,f in itertools.chain(*feature_generators):
            if self.feat_index.has_key(f):
                F[i,self.feat_index[f]] = 1
        return F

    def fit_transform(self, candidates):
        """Assembles the set of features to be used, and applies this transformation to the candidates"""
        feature_generators = self._match_contexts(self._preprocess_candidates(candidates))

        # Assemble and return the sparse feature matrix
        f_index = defaultdict(list)
        for i,f in itertools.chain(*feature_generators):
            f_index[f].append(i)

        # Assemble and return sparse feature matrix
        # Also assemble reverse index of feature matrix index -> feature verbose name
        self.feat_index     = {}
        self.feat_inv_index = {}
        F                   = sparse.lil_matrix((len(candidates), len(f_index.keys())))
        for j,f in enumerate(f_index.keys()):
            self.feat_index[f] = j
            self.feat_inv_index[j] = f
            for i in f_index[f]:
                F[i,j] = 1
        return F

    def get_features_by_candidate(self, candidate):
        feature_generators = self._match_contexts(self._preprocess_candidates([candidate]))
        feats = []
        for i,f in itertools.chain(*feature_generators):
            feats.append(f)
        return feats

class NgramFeaturizer(Featurizer):
    """Feature for relations (of arity >= 1) defined over Ngram objects."""
    def _preprocess_candidates(self, candidates):
        # for c in candidates:
            # if not isinstance(c.context, dict):
            #     c.context = get_as_dict(c.context)
            # if c.context['xmltree'] is None:
            #     c.context['xmltree'] = corenlp_to_xmltree(c.context)
        return candidates

    def _match_contexts(self, candidates):
        feature_generators = []

        # Unary relations
        if self.arity == 1:

            # Add DDLIB entity features
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_ddlib_feats(c, range(c.get_word_start(), c.get_word_end()+1)), 'DDLIB_', candidates))

            # Add TreeDLib entity features
            # get_feats = compile_entity_feature_generator()
            # feature_generators.append(self._generate_context_feats( \
            #     lambda c : get_feats(c.context['xmltree'].root, range(c.word_start, c.word_end+1)), 'TDL_', candidates))

        if self.arity == 2:
            raise NotImplementedError("Featurizer needs to be implemented for binary relations!")
        return feature_generators


class TableNgramFeaturizer(NgramFeaturizer):
    """In addition to Ngram features, add table structure information."""
    def _match_contexts(self, candidates):
        feature_generators = super(TableNgramFeaturizer, self)._match_contexts(candidates)

        # Unary relations
        if self.arity == 1:
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_table_feats(c), 'TABLE_', candidates))

        if self.arity == 2:
            raise NotImplementedError("Featurizer needs to be implemented for binary relations!")

        return feature_generators


class LegacyCandidateFeaturizer(Featurizer):
    """Temporary class to handle v0.2 Candidate objects."""
    def _match_contexts(self, candidates):
        feature_generators = []

        # Unary relations
        if self.arity == 1:

            # Add DDLIB entity features
            feature_generators.append(self._generate_context_feats( \
                lambda c : get_ddlib_feats(c, c.idxs), 'DDLIB_', candidates))

            # Add TreeDLib entity features
            if candidates[0].root is not None:
                get_feats = compile_entity_feature_generator()
                feature_generators.append(self._generate_context_feats( \
                    lambda c : get_feats(c.root, c.idxs), 'TDLIB_', candidates))

        if self.arity == 2:

            # Add TreeDLib relation features
            if candidates[0].root is not None:
                get_feats = compile_relation_feature_generator()
                feature_generators.append(self._generate_context_feats( \
                    lambda c : get_feats(c.root, c.e1_idxs, c.e2_idxs), 'TDLIB_', candidates))
        return feature_generators

# <<<<<<< HEAD
# from collections import defaultdict
# import scipy.sparse as sparse

# from lxml import etree as et
# from entity_features import get_ddlib_feats
# from entity_features import compile_entity_feature_generator

# class Featurizer(object):
#     def __init__(self, candidates, corpus):
#         self.number_of_candidates = len(candidates)
#         self._features_by_id = defaultdict(list)
#         print "Extracting features..."
#         self.feats = self.extract_features(candidates, corpus)
#         print "Extracted {} features for each of {} candidates".format(self.num_features(), self.num_candidates())

#     def num_candidates(self):
#         return self.number_of_candidates

#     def num_features(self):
#         return self.feats.shape[1]

#     def extract_features(self, candidates, corpus):
#         f_index = self._get_feature_index(candidates, corpus)
#         f_matrix = self._get_feature_matrix(f_index)
#         return f_matrix

#     def _get_feature_index(self, candidates, corpus):
#         f_index = defaultdict(list)
#         for j,cand in enumerate(candidates):
#             for feat in self._featurize(cand):
#                 self._features_by_id[cand.id].append(feat)
#                 f_index[feat].append(j)
#         return f_index

#     def _featurize(self):
#         raise NotImplementedError

#     def _get_feature_matrix(self, f_index):
#         # Apply the feature generator, constructing a sparse matrix incrementally
#         # Note that lil_matrix should be relatively efficient as we proceed row-wise
#         self.feats = sparse.lil_matrix((self.num_candidates(), len(f_index)))
#         for j,feat in enumerate(f_index.keys()):
#             for i in f_index[feat]:
#                 self.feats[i,j] = 1
#         return self.feats

#     def get_features_by_id(self, id):
#         features = self._features_by_id[id]
#         return features if features is not None else None

#     def get_features(self):
#         return self.feats

# class NgramFeaturizer(Featurizer):
#     def _featurize(self, cand):
#         # This is a poor man's substitue for coreNLP until they come together
#         for feat in self.generate_temp_nlp_feats(cand):
#             yield feat
#         # for feat in self.generate_nlp_feats(cand, context):
#         #     yield feat
#         # for feat in self.generate_ddlib_feats(cand, context):
#         #     yield feat

#     def generate_temp_nlp_feats(self, cand):
#         for ngram in self.get_ngrams(cand.get_attrib_tokens('words')):
#             yield ''.join(["BASIC_NGRAM_", ngram])

#     def get_ngrams(self, words, n_max=3):
#         N = len(words)
#         for root in range(N):
#             for n in range(min(n_max, N - root)):
#                 yield '_'.join(words[root:root+n+1])

#     # def generate_nlp_feats(self, cand, context):
#     #     get_nlp_feats = compile_entity_feature_generator()
#     #     for feat in get_nlp_feats(cand.root, cand.idxs):
#     #         yield ''.join(["NLP_", feat])

#     # def generate_ddlib_feats(self, cand, context):
#     #     for feat in get_ddlib_feats(cand, cand.idxs):
#     #         yield ''.join(["DDLIB_", feat])

# class TableNgramFeaturizer(NgramFeaturizer):
#     def _featurize(self, cand):
#         for feat in super(TableNgramFeaturizer, self)._featurize(cand):
#             yield feat
#         for feat in self.generate_table_feats(cand):
#             yield ''.join(["TABLE_",feat])

#     def generate_table_feats(self, cand):
#         yield "ROW_NUM_%s" % cand.context.row_num
#         yield "COL_NUM_%s" % cand.context.col_num
#         yield "HTML_TAG_" + cand.context.html_tag
#         for attr in cand.context.html_attrs:
#             yield "HTML_ATTR_" + attr
#         for tag in cand.context.html_anc_tags:
#             yield "HTML_ANC_TAG_" + tag
#         for attr in cand.context.html_anc_attrs:
#             yield "HTML_ANC_ATTR_" + attr
#         for ngram in self.get_aligned_ngrams(cand, axis='row'):
#             yield "ROW_NGRAM_" + ngram
#         for ngram in self.get_aligned_ngrams(cand, axis='col'):
#             yield "COL_NGRAM_" + ngram

#     # NOTE: it may just be simpler to search by row_num, col_num rather than
#     # traversing tree, though other range features may benefit from tree structure
#     def get_aligned_ngrams(self, cand, n_max=3, attribute='words', axis='row'):
#         # SQL join method (eventually)
#         if axis=='row':
#             aligned_phrases = [phrase for phrase in cand.context.table.phrases if phrase.row_num == cand.context.row_num]
#         elif axis=='col':
#             aligned_phrases = [phrase for phrase in cand.context.table.phrases if phrase.col_num == cand.context.col_num]
#         for phrase in aligned_phrases:
#             words = phrase.words
#             for ngram in self.get_ngrams(words, n_max=n_max):
#                 yield ngram
#         # Tree traversal method:
#         # root = et.fromstring(context.html)
#         # if axis=='row':
#             # snorkel_ids = root.xpath('//*[@snorkel_id="%s"]/following-sibling::*/@snorkel_id' % cand.cell_id)
#         # if axis=='col':
#             # position = len(root.xpath('//*[@snorkel_id="%s"]/following-sibling::*/@snorkel_id' % cand.cell_id)) + 1
#             # snorkel_ids = root.xpath('//*[@snorkel_id][position()=%d]/@snorkel_id' % position)
# =======

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import os
import re
import sys

sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))

from collections import defaultdict
from snorkel.features.entity_features import compile_entity_feature_generator, get_ddlib_feats
from functools import partial
from snorkel.models import Span
from snorkel.utils import get_as_dict
from string import punctuation
from tree_structs import corenlp_to_xmltree
from treedlib import compile_relation_feature_generator


def get_span_splits(candidate, stopwords=None):
    """Base function for candidate span tokens split on whitespace
    and punctuation
    candidate: @Candidate to extract features for
    stopwords: @set of stopwords to filter out from tokens
    """
    split_pattern = r'[\s{}]+'.format(re.escape(punctuation))
    for i, arg in enumerate(candidate.get_contexts()):
        for token in re.split(split_pattern, arg.get_span().lower()):
            if stopwords is None or token not in stopwords:
                yield 'SPAN_SPLIT[{0}][{1}]'.format(i, token), 1


def get_span_splits_stopwords(stopwords):
    """Get a span split unary function"""
    return partial(get_span_splits, stopwords=stopwords)


def get_unary_span_feats(sidxs, sentence, stopwords):
    """Get unary span features from DDLib and TreeDLib"""
    get_tdl_feats = compile_entity_feature_generator()
    sent_dict     = get_as_dict(sentence)
    xmltree       = corenlp_to_xmltree(sent_dict)
    if len(sidxs) > 0:
        # Add DDLIB entity features
        for f in get_ddlib_feats(sent_dict, sidxs):
            yield 'DDL_' + f, 1
        # Add TreeDLib entity features
        for f in get_tdl_feats(xmltree.root, sidxs, stopwords=stopwords):
            yield 'TDL_' + f, 1


def get_binary_span_feats(sidxs, sentence, stopwords):
    """Get binary (relation) span features from TreeDLib"""
    get_tdl_feats = compile_relation_feature_generator()
    xmltree = corenlp_to_xmltree(get_as_dict(sentence))
    s1_idxs, s2_idxs = sidxs
    if len(s1_idxs) > 0 and len(s2_idxs) > 0:
        # Apply TDL features
        for f in get_tdl_feats(xmltree.root, s1_idxs, s2_idxs,
                               stopwords=stopwords):
            yield 'TDL_' + f, 1


def get_span_feats(candidate, stopwords=None):
    """Base function for sentence dependency path features
    candidate: @Candidate to extract features for
    stopwords: @set of stopwords to filter out from dependency path
    """
    args = candidate.get_contexts()
    if not isinstance(args[0], Span):
        raise ValueError("Accepts Span-type arguments, %s-type found.")
    # Unary candidates
    if len(args) == 1:
        sidxs = list(range(args[0].get_word_start(), args[0].get_word_end() + 1))
        return get_unary_span_feats(sidxs, candidate.get_parent(), stopwords)
    # Binary candidates
    elif len(args) == 2:
        sidxs = [list(range(a.get_word_start(), a.get_word_end() + 1)) for a in args]
        return get_binary_span_feats(sidxs, candidate.get_parent(), stopwords)
    else:
        raise NotImplementedError("Only handles unary or binary candidates")


def get_span_feats_stopwords(stopwords):
    """Get a span dependency tree unary function"""
    return partial(get_span_feats, stopwords=stopwords)


def get_entity_word_idxs(sentence, et, cid):
    """Get indices of @sentence tokens with type @et and id @cid"""
    itr = enumerate(zip(sentence.entity_types, sentence.entity_cids))
    return [i for i, (t, c) in itr if c == cid and t == et]


def get_first_document_span_feats(candidate, stopwords=None):
    """Base function for sentence dependency path features where the sentence
       is the first sentence in the parent @Document of @candidate which
       contains the entities in @candidate
    candidate: @Candidate to extract features for
    stopwords: @set of stopwords to filter out from dependency path
    """
    # Get candidate entity types and cids
    entity_types = [
        c.get_attrib_tokens('entity_types')[0] for c in candidate.get_contexts()
    ]
    #entity_cids = candidate.get_cids()
    entity_cids = [
        c.get_attrib_tokens('entity_cids')[0] for c in candidate.get_contexts()
    ]
    # Look for entity mentions in each sentence
    for sentence in candidate.get_parent().document.get_sentence_generator():
        mention_idxs = [
            get_entity_word_idxs(sentence, t, cid)
            for t, cid in zip(entity_types, entity_cids)
        ]
        if all(len(idxs) > 0 for idxs in mention_idxs):
            break
    # Get features for first valid sentence
    if all(len(idxs) > 0 for idxs in mention_idxs):
        if len(mention_idxs) == 1:
            return get_unary_span_feats(mention_idxs[0], sentence, stopwords)
        elif len(mention_idxs) == 2:
            return get_binary_span_feats(mention_idxs, sentence, stopwords)
        else:
            raise NotImplementedError("Only handles unary or binary candidates")


def get_first_document_span_feats_stopwords(stopwords):
    """Get a first document span dependency tree unary function"""
    return partial(get_first_document_span_feats, stopwords=stopwords)


def get_entity_type_counts(context, entity_types):
    """Get count of entity cids in @context by entity_type in @entity_types"""
    type_counts = {et: defaultdict(int) for et in entity_types}
    for sentence in context.get_sentence_generator():
        cur_et, cur_cid = None, None
        # Iterate over entities in sentence
        for et, cid in zip(sentence.entity_types, sentence.entity_cids):
            # If current entity changes, add to its count
            if et != cur_et or cid != cur_cid:
                if cur_et in type_counts:
                    type_counts[cur_et][cur_cid] += 1
            cur_et, cur_cid = et, cid
        # Add last to count
        if cur_et in type_counts:
            type_counts[cur_et][cur_cid] += 1
    return type_counts


def get_relative_frequency_feats(candidate, context):
    """Base getting relative frequency of @candidate entities among 
       entities in @context
    candidate: @Candidate to extract features for
    context: @Context over which to get relative frequency
    """
    # Get candidate entity types and cids
    entity_types = [
        c.get_attrib_tokens('entity_types')[0] for c in candidate.get_contexts()
    ]
    #entity_cids = candidate.get_cids()
    entity_cids = [
        c.get_attrib_tokens('entity_cids')[0] for c in candidate.get_contexts()
    ]
    # Get counts for entities of relevant types
    type_counts = get_entity_type_counts(context, entity_types)
    # Get most frequent entities and counts for candidate entities
    max_counts = {
        et: max(1, max(type_counts[et].values())) for et in entity_types
    }
    entity_counts = {
        cid: type_counts[et][cid] for et, cid in zip(entity_types, entity_cids)
    }
    # Compute relative frequency
    for i, (cid, et) in enumerate(zip(entity_cids, entity_types)):
        p = float(entity_counts[cid]) / max_counts[et]
        yield "ENTITY_RELATIVE_FREQUENCY[{0}]".format(i), p


def get_document_relative_frequency_feats(candidate):
    """Apply @get_relative_frequency_feats over the parent
       @Document of @candidate
    """
    doc = candidate.get_parent().document
    return get_relative_frequency_feats(candidate, doc)


def get_sentence_relative_frequency_feats(candidate):
    """Apply @get_relative_frequency_feats over the parent
       @Sentence of @candidate
    """
    sentence = candidate.get_parent()
    return get_relative_frequency_feats(candidate, sentence)

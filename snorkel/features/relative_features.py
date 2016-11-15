import os
import sys

sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))

from .models import Span
from collections import defaultdict
from entity_features import compile_entity_feature_generator, get_ddlib_feats
from functools import partial
from string import punctuation
from tree_structs import corenlp_to_xmltree
from treedlib import compile_relation_feature_generator
from utils import get_as_dict


def get_span_splits(candidate, stopwords=None):
    split_pattern = r'[\s{}]+'.format(re.escape(punctuation))
    for i, arg in enumerate(candidate.get_contexts()):
        for token in re.split(split_pattern, s.get_span().lower()):
            if stopwords is None or token not in stopwords:
                yield 'SPAN_SPLIT[{0}][{1}]'.format(i, token), 1


def get_span_splits_stopwords(stopwords):
    return partial(get_span_splits, stopwords=stopwords)


def get_span_feats(candidate, stopwords=None):
    args = candidate.get_contexts()
    if not isinstance(args[0], Span):
        raise ValueError("Accepts Span-type arguments, %s-type found.")

    # Unary candidates
    if len(args) == 1:
        get_tdl_feats = compile_entity_feature_generator()
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

    # Binary candidates
    elif len(args) == 2:
        get_tdl_feats = compile_relation_feature_generator()
        span1, span2  = args
        xmltree       = corenlp_to_xmltree(get_as_dict(span1.parent))
        s1_idxs       = range(span1.get_word_start(), span1.get_word_end() + 1)
        s2_idxs       = range(span2.get_word_start(), span2.get_word_end() + 1)
        if len(s1_idxs) > 0 and len(s2_idxs) > 0:

            # Apply TDL features
            for f in get_tdl_feats(xmltree.root, s1_idxs, s2_idxs,
                                   stopwords=stopwords):
                yield 'TDL_' + f, 1
    else:
        raise NotImplementedError("Only handles unary or binary candidates")


def get_span_feats_stopwords(stopwords):
    return partial(get_span_feats, stopwords=stopwords)


def get_entity_word_idxs(sentence, canonical_id):
    pass


def get_first_document_span_feats(candidate, stopwords=None):
    canonical_ids = candidate.get_canonical_ids()
    for sentence in candidate.get_parent().parent.get_sentence_generator():
        mention_idxs = [
            get_entity_word_idxs(sentence, cid) for cid in canonical_ids
        ]
        if all(len(idxs) > 0 for idxs in mention_idxs):
            break
    if all(len(idxs) > 0 for idxs in mention_idxs):
        if len(mention_idxs) == 1:
            get_tdl_feats = compile_entity_feature_generator()
            span          = args[0]
            sent_dict     = get_as_dict(sentence)
            xmltree       = corenlp_to_xmltree(sent_dict)
            for f in get_ddlib_feats(sent_dict, mention_idxs[0]):
                yield 'DDL_' + f, 1
            for f in get_tdl_feats(xmltree.root, mention_idxs[0]):
                yield 'TDL_' + f, 1
        elif len(mention_idxs) == 2:
            get_tdl_feats = compile_relation_feature_generator()
            xmltree       = corenlp_to_xmltree(get_as_dict(sentence))
            for f in get_tdl_feats(xmltree.root, mention_idxs[0], 
                                   mention_idxs[1], stopwords=stopwords):
                yield 'TDL_' + f, 1
        else:
            raise NotImplementedError("Only handles unary or binary candidates")


def get_first_document_span_feats_stopwords(stopwords):
    return partial(get_first_document_span_feats, stopwords=stopwords)


def get_entity_type_max_counts(context):
    counts = defaultdict(int)
    pass


def get_entity_counts(canonical_ids, context):
    counts = {cid: 0 for cid in canonical_ids}
    pass


def get_relative_frequency_feats(candidate, context):
    max_counts = get_entity_type_max_counts(context)
    canonical_ids = candidate.get_canonical_ids()
    entity_counts = get_entity_counts(canonical_ids, context)
    for i, (ct, max_ct) in enumerate(zip(entity_counts, max_counts)):
        yield "ENTITY_RELATIVE_FREQUENCY[{0}]".format(i), float(ct) / max_ct


def get_document_relative_frequency_feats(candidate):
    doc = candidate.get_parent().parent
    return get_relative_frequency_feats(candidate, doc)


def get_sentence_relative_frequency_feats(candidate):
    sentence = candidate.get_parent()
    return get_relative_frequency_feats(candidate, sentence)

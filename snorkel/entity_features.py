import sys, os
sys.path.append(os.environ['SNORKELHOME'] + '/treedlib/treedlib')
from templates import *
from lf_helpers import get_row_ngrams, get_col_ngrams, get_neighbor_cell_ngrams
from utils import get_as_dict

def compile_entity_feature_generator():
  """
  Given optional arguments, returns a generator function which accepts an xml root
  and a list of indexes for a mention, and will generate relation features for this entity
  """

  BASIC_ATTRIBS_REL = ['lemma', 'dep_label']

  m = Mention(0)

  # Basic relation feature templates
  temps = [
    [Indicator(m, a) for a in BASIC_ATTRIBS_REL],
    Indicator(m, 'dep_label,lemma'),
    # The *first element on the* path to the root: ngram lemmas along it
    Ngrams(Parents(m, 3), 'lemma', (1,3)),
    Ngrams(Children(m), 'lemma', (1,3)),
    # The siblings of the mention
    [LeftNgrams(LeftSiblings(m), a) for a in BASIC_ATTRIBS_REL],
    [RightNgrams(RightSiblings(m), a) for a in BASIC_ATTRIBS_REL]
  ]

  # return generator function
  return Compile(temps).apply_mention

def get_ddlib_feats(context, idxs):
  """
  Minimalist port of generic mention features from ddlib
  """
  for seq_feat in _get_seq_features(context, idxs):
    yield seq_feat

  for window_feat in _get_window_features(context, idxs):
    yield window_feat

  if context['words'][idxs[0]][0].isupper():
      yield "STARTS_WITH_CAPITAL"

  yield "LENGTH_{}".format(len(idxs))

def _get_seq_features(context, idxs):
  yield "WORD_SEQ_[" + " ".join(context['words'][i] for i in idxs) + "]"
  yield "LEMMA_SEQ_[" + " ".join(context['lemmas'][i] for i in idxs) + "]"
  yield "POS_SEQ_[" + " ".join(context['pos_tags'][i] for i in idxs) + "]"
  yield "DEP_SEQ_[" + " ".join(context['dep_labels'][i] for i in idxs) + "]"

def _get_window_features(context, idxs, window=3, combinations=True, isolated=True):
    left_lemmas = []
    left_pos_tags = []
    right_lemmas = []
    right_pos_tags = []
    try:
        for i in range(1, window + 1):
            lemma = context['lemmas'][idxs[0] - i]
            try:
                float(lemma)
                lemma = "_NUMBER"
            except ValueError:
                pass
            left_lemmas.append(lemma)
            left_pos_tags.append(context['pos_tags'][idxs[0] - i])
    except IndexError:
        pass
    left_lemmas.reverse()
    left_pos_tags.reverse()
    try:
        for i in range(1, window + 1):
            lemma = context['lemmas'][idxs[-1] + i]
            try:
                float(lemma)
                lemma = "_NUMBER"
            except ValueError:
                pass
            right_lemmas.append(lemma)
            right_pos_tags.append(context['pos_tags'][idxs[-1] + i])
    except IndexError:
        pass
    if isolated:
        for i in range(len(left_lemmas)):
            yield "W_LEFT_" + str(i+1) + "_[" + " ".join(left_lemmas[-i-1:]) + \
                "]"
            yield "W_LEFT_POS_" + str(i+1) + "_[" + " ".join(left_pos_tags[-i-1:]) +\
                "]"
        for i in range(len(right_lemmas)):
            yield "W_RIGHT_" + str(i+1) + "_[" + " ".join(right_lemmas[:i+1]) +\
                "]"
            yield "W_RIGHT_POS_" + str(i+1) + "_[" + \
                " ".join(right_pos_tags[:i+1]) + "]"
    if combinations:
        for i in range(len(left_lemmas)):
            curr_left_lemmas = " ".join(left_lemmas[-i-1:])
            try:
                curr_left_pos_tags = " ".join(left_pos_tags[-i-1:])
            except TypeError:
                new_pos_tags = []
                for pos in left_pos_tags[-i-1:]:
                    to_add = pos
                    if not to_add:
                        to_add = "None"
                    new_pos_tags.append(to_add)
                curr_left_pos_tags = " ".join(new_pos_tags)
            for j in range(len(right_lemmas)):
                curr_right_lemmas = " ".join(right_lemmas[:j+1])
                try:
                    curr_right_pos_tags = " ".join(right_pos_tags[:j+1])
                except TypeError:
                    new_pos_tags = []
                    for pos in right_pos_tags[:j+1]:
                        to_add = pos
                        if not to_add:
                            to_add = "None"
                        new_pos_tags.append(to_add)
                    curr_right_pos_tags = " ".join(new_pos_tags)
                yield "W_LEMMA_L_" + str(i+1) + "_R_" + str(j+1) + "_[" + \
                    curr_left_lemmas + "]_[" + curr_right_lemmas + "]"
                yield "W_POS_L_" + str(i+1) + "_R_" + str(j+1) + "_[" + \
                    curr_left_pos_tags + "]_[" + curr_right_pos_tags + "]"

def tabledlib_unary_features(span):
    phrase = span.parent
    yield u"HTML_TAG_" + phrase.html_tag
    for attr in phrase.html_attrs:
        yield u"HTML_ATTR_[" + attr + "]"
    for tag in phrase.html_anc_tags:
        yield u"HTML_ANC_TAG_[" + tag + "]"
    # for attr in phrase.html_anc_attrs:
        # yield u"HTML_ANC_ATTR_[" + attr + "]"
    if phrase.row_num is not None and phrase.col_num is not None:
        yield u"ROW_NUM_[%s]" % phrase.row_num
        yield u"COL_NUM_[%s]" % phrase.col_num
        for attrib in ['lemmas']: #,'words', 'pos_tags', 'ner_tags']:
            for ngram in get_row_ngrams(span, n_max=2, attrib=attrib):
                yield "ROW_%s_[%s]" % (attrib.upper(), ngram)
                if attrib=="lemmas":
                    try:
                        if float(ngram).is_integer():
                            yield u"ROW_INT"
                        else:
                            yield u"ROW_FLOAT"
                    except:
                        pass
            for ngram in get_col_ngrams(span, n_max=2, attrib=attrib):
                yield "COL_%s_[%s]" % (attrib.upper(), ngram)
                if attrib=="lemmas":
                    try:
                        if float(ngram).is_integer():
                            yield u"COL_INT"
                        else:
                            yield u"COL_FLOAT"
                    except:
                        pass
            for ngram in get_row_ngrams(span, n_max=2, attrib=attrib, direct=False, infer=True):
                yield "ROW_INFERRED_%s_[%s]" % (attrib.upper(), ngram)         
            for ngram in get_col_ngrams(span, n_max=2, attrib=attrib, direct=False, infer=True):
                yield "COL_INFERRED_%s_[%s]" % (attrib.upper(), ngram)         
            # for (ngram, direction) in get_neighbor_cell_ngrams(span, dist=2, directions=True, n_max=3, attrib=attrib):
            #     yield "NEIGHBOR_%s_%s_[%s]" % (direction, attrib.upper(), ngram)
            #     if attrib=="lemmas":
            #         try:
            #             if float(ngram).is_integer():
            #                 yield "NEIGHBOR_%s_INT" % side
            #             else:
            #                 yield "NEIGHBOR_%s_FLOAT" % side
            #         except:
            #             pass

def tabledlib_binary_features(span1, span2, s1_idxs, s2_idxs):
    for feat in get_ddlib_feats(get_as_dict(span1.parent), s1_idxs):
        yield "e1_" + feat
    for feat in tabledlib_unary_features(span1):
        yield "e1_" + feat
    for feat in get_ddlib_feats(get_as_dict(span2.parent), s2_idxs):
        yield "e2_" + feat
    for feat in tabledlib_unary_features(span2):
        yield "e2_" + feat
    if span1.parent.table is not None and span2.parent.table is not None:
        if span1.parent.table == span2.parent.table:
            yield u"SAME_TABLE"
            if span1.parent.cell is not None and span2.parent.cell is not None:
                row_diff = span1.parent.row_num - span2.parent.row_num
                col_diff = span1.parent.col_num - span2.parent.col_num
                yield u"SAME_TABLE_ROW_DIFF_[%s]" % row_diff
                yield u"SAME_TABLE_COL_DIFF_[%s]" % col_diff
                yield u"SAME_TABLE_MANHATTAN_DIST_[%s]" % str(abs(row_diff) + abs(col_diff))
                if span1.parent.cell == span2.parent.cell:
                    yield u"SAME_CELL"
                    yield u"WORD_DIFF_[%s]" % (span1.get_word_start() - span2.get_word_start())
                    yield u"CHAR_DIFF_[%s]" % (span1.char_start - span2.char_start)
                    if span1.parent == span2.parent:
                        yield u"SAME_PHRASE"
        else:
            if span1.parent.cell is not None and span2.parent.cell is not None:
                row_diff = span1.parent.row_num - span2.parent.row_num
                col_diff = span1.parent.col_num - span2.parent.col_num
                yield u"DIFF_TABLE_ROW_DIFF_[%s]" % row_diff
                yield u"DIFF_TABLE_COL_DIFF_[%s]" % col_diff
                yield u"DIFF_TABLE_MANHATTAN_DIST_[%s]" % str(abs(row_diff) + abs(col_diff))

import sys
import os

# from templates import *
from lf_helpers import *
from utils import get_as_dict
from .models import ImplicitSpan

sys.path.append(os.environ['SNORKELHOME'] + '/treedlib/treedlib')

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
    yield "SPAN_TYPE_[%s]" % ('IMPLICIT' if isinstance(span, ImplicitSpan) else 'EXPLICIT') 
    phrase = span.parent
    yield u"HTML_TAG_" + phrase.html_tag
    # for attr in phrase.html_attrs:
    #     yield u"HTML_ATTR_[" + attr + "]"
    for tag in phrase.html_anc_tags:
        yield u"HTML_ANC_TAG_[" + tag + "]"
    # for attr in phrase.html_anc_attrs:
        # yield u"HTML_ANC_ATTR_[" + attr + "]"
    for attrib in ['words']: #,'lemmas', 'pos_tags', 'ner_tags']:
        for ngram in span.get_attrib_tokens(attrib):
            yield "CONTAINS_%s_[%s]" % (attrib.upper(), ngram)
        for ngram in get_left_ngrams(span, window=7, n_max=2, attrib=attrib):
            yield "LEFT_%s_[%s]" % (attrib.upper(), ngram)
        for ngram in get_right_ngrams(span, window=7, n_max=2, attrib=attrib):
            yield "RIGHT_%s_[%s]" % (attrib.upper(), ngram)
        if phrase.row_num is None or phrase.col_num is None:
            for ngram in get_neighbor_phrase_ngrams(span, d=1, n_max=2, attrib=attrib):
                yield "NEIGHBOR_PHRASE_%s_[%s]" % (attrib.upper(), ngram)
        else:
            for ngram in get_cell_ngrams(span, n_max=2, attrib=attrib):
                yield "CELL_%s_[%s]" % (attrib.upper(), ngram)
            yield u"ROW_NUM_[%s]" % phrase.row_num
            yield u"COL_NUM_[%s]" % phrase.col_num
            for axis in ['row', 'col']:
                for ngram in get_head_ngrams(span, axis, n_max=2, attrib=attrib):
                    yield "%s_HEAD_%s_[%s]" % (axis.upper(), attrib.upper(), ngram)
            for ngram in get_row_ngrams(span, n_max=2, attrib=attrib):
                yield "ROW_%s_[%s]" % (attrib.upper(), ngram)
            for ngram in get_col_ngrams(span, n_max=2, attrib=attrib):
                yield "COL_%s_[%s]" % (attrib.upper(), ngram)
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
    # for feat in get_ddlib_feats(get_as_dict(span1.parent), s1_idxs):
    #     yield "DDL_e1_" + feat
    for feat in tabledlib_unary_features(span1):
        yield "e1_" + feat
    # for feat in get_ddlib_feats(get_as_dict(span2.parent), s2_idxs):
    #     yield "DDL_e2_" + feat
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

_Bbox = namedtuple('bbox', ['top','bottom','left','right'], verbose = False)
def _bbox_from_span(span):
    if hasattr(span, 'top') and span.top is not None:
        return _Bbox(min(span.get_attrib_tokens('top')), 
                    max(span.get_attrib_tokens('bottom')),
                    min(span.get_attrib_tokens('left')),
                    max(span.get_attrib_tokens('right')))
    else:
        return None

def visual_binary_features(span1, span2, s1_idxs = None, s2_idxs = None):
    '''
    Features about the relative positioning of two spans
    '''
    # Skip when coordinates are not available
    
    bbox1 = _bbox_from_span(span1)
    bbox2 = _bbox_from_span(span2)
    
    if not (bbox1 and bbox2): return
    yield 'HAS_COORDS'
    
    if bbox1.top < bbox2.bottom and bbox2.top < bbox1.bottom:
        yield 'Y_ALIGNED'
    
    v_aligned = False
    if abs(bbox1.left - bbox2.left) < 1:
        v_aligned = True
        yield 'LEFT_ALIGNED'
        
    if abs(bbox1.right - bbox2.right) < 1:
        v_aligned = True
        yield 'RIGHT_ALIGNED'
        
    if abs((bbox1.left+bbox1.right)/2 - (bbox2.left+bbox2.right)/2) < 1:
        v_aligned = True
        yield 'CENTER_ALIGNED'

    if v_aligned: yield 'X_ALIGNED'

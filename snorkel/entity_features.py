import sys, os
sys.path.append(os.environ['SNORKELHOME'] + '/treedlib/treedlib')
from templates import *

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

def get_ddlib_feats(cand, idxs):
  """
  Minimalist port of generic mention features from ddlib
  """
  for seq_feat in _get_seq_features(cand, idxs):
    yield seq_feat

  for window_feat in _get_window_features(cand, idxs):
    yield window_feat

  if cand.context.words[idxs[0]][0].isupper():
    yield u"STARTS_WITH_CAPITAL"

  yield u"NUM_WORDS_%s" % len(idxs)

def _get_seq_features(cand, idxs):
    yield "WORD_SEQ_[" + " ".join(cand.context.words[i] for i in idxs) + "]"
    yield "LEMMA_SEQ_[" + " ".join(cand.context.lemmas[i] for i in idxs) + "]"
    yield "POS_SEQ_[" + " ".join(cand.context.poses[i] for i in idxs) + "]"
    yield "DEP_SEQ_[" + " ".join(cand.context.dep_labels[i] for i in idxs) + "]"

def _get_window_features(cand, idxs, window=3, combinations=True, isolated=True):
    left_lemmas = []
    left_poses = []
    right_lemmas = []
    right_poses = []
    try:
        for i in range(1, window + 1):
            lemma = cand.context.lemmas[idxs[0] - i]
            try:
                float(lemma)
                lemma = "_NUMBER"
            except ValueError:
                pass
            left_lemmas.append(lemma)
            left_poses.append(cand.context.poses[idxs[0] - i])
    except IndexError:
        pass
    left_lemmas.reverse()
    left_poses.reverse()
    try:
        for i in range(1, window + 1):
            lemma = cand.context.lemmas[idxs[-1] + i]
            try:
                float(lemma)
                lemma = "_NUMBER"
            except ValueError:
                pass
            right_lemmas.append(lemma)
            right_poses.append(cand.context.poses[idxs[-1] + i])
    except IndexError:
        pass
    if isolated:
        for i in range(len(left_lemmas)):
            yield "W_LEFT_" + str(i+1) + "_[" + " ".join(left_lemmas[-i-1:]) + \
                "]"
            yield "W_LEFT_POS_" + str(i+1) + "_[" + " ".join(left_poses[-i-1:]) +\
                "]"
        for i in range(len(right_lemmas)):
            yield "W_RIGHT_" + str(i+1) + "_[" + " ".join(right_lemmas[:i+1]) +\
                "]"
            yield "W_RIGHT_POS_" + str(i+1) + "_[" + \
                " ".join(right_poses[:i+1]) + "]"
    if combinations:
        for i in range(len(left_lemmas)):
            curr_left_lemmas = " ".join(left_lemmas[-i-1:])
            try:
                curr_left_poses = " ".join(left_poses[-i-1:])
            except TypeError:
                new_poses = []
                for pos in left_poses[-i-1:]:
                    to_add = pos
                    if not to_add:
                        to_add = "None"
                    new_poses.append(to_add)
                curr_left_poses = " ".join(new_poses)
            for j in range(len(right_lemmas)):
                curr_right_lemmas = " ".join(right_lemmas[:j+1])
                try:
                    curr_right_poses = " ".join(right_poses[:j+1])
                except TypeError:
                    new_poses = []
                    for pos in right_poses[:j+1]:
                        to_add = pos
                        if not to_add:
                            to_add = "None"
                        new_poses.append(to_add)
                    curr_right_poses = " ".join(new_poses)
                yield "W_LEMMA_L_" + str(i+1) + "_R_" + str(j+1) + "_[" + \
                    curr_left_lemmas + "]_[" + curr_right_lemmas + "]"
                yield "W_POS_L_" + str(i+1) + "_R_" + str(j+1) + "_[" + \
                    curr_left_poses + "]_[" + curr_right_poses + "]"

def get_table_feats(cand):
    yield u"ROW_NUM_[%s]" % cand.context.row_num
    yield u"COL_NUM_[%s]" % cand.context.col_num
    yield u"HTML_TAG_" + cand.context.html_tag
    for attr in cand.context.html_attrs:
        yield u"HTML_ATTR_" + attr
    for tag in cand.context.html_anc_tags:
        yield u"HTML_ANC_TAG_" + tag
    for attr in cand.context.html_anc_attrs:
        yield u"HTML_ANC_ATTR_" + attr
    for attr in ['words']: # ['lemmas','poses']
        for ngram in cand.row_ngrams(attr=attr):
            yield "ROW_%s_%s" % (attr.upper(), ngram)
            if attr=="lemmas":
                try:
                    if float(ngram).is_integer():
                        yield u"ROW_INT"
                    else:
                        yield u"ROW_FLOAT"
                except:
                    pass
        for ngram in cand.col_ngrams(attr=attr):
            yield "COL_%s_%s" % (attr.upper(), ngram)
            if attr=="lemmas":
                try:
                    if float(ngram).is_integer():
                        yield u"COL_INT"
                    else:
                        yield u"COL_FLOAT"
                except:
                    pass
        for (ngram, side) in cand.neighbor_ngrams(attr=attr):
            yield "NEIGHBOR_%s_%s_%s" % (side, attr.upper(), ngram)
            if attr=="lemmas":
                try:
                    if float(ngram).is_integer():
                        yield "NEIGHBOR_%s_INT" % side
                    else:
                        yield "NEIGHBOR_%s_FLOAT" % side
                except:
                    pass

def get_relation_table_feats(cand):
    if cand.ngram0.context.table == cand.ngram1.context.table:
        yield u"SAME_TABLE"
        row_diff = cand.ngram0.context.row_num - cand.ngram1.context.row_num
        yield u"ROW_DIFF_[%s]" % row_diff
        col_diff = cand.ngram0.context.col_num - cand.ngram1.context.col_num
        yield u"COL_DIFF_[%s]" % col_diff
        yield u"MANHATTAN_DIST_[%s]" % str(abs(row_diff) + abs(col_diff))
        if cand.ngram0.context.cell == cand.ngram1.context.cell:
            yield u"SAME_CELL"
            yield u"WORD_DIFF_[%s]" % cand.ngram0.get_word_start() - cand.ngram1.get_word_start()
            yield u"CHAR_DIFF_[%s]" % cand.ngram0.char_start - cand.ngram1.char_start
            if cand.ngram0.context.phrase == cand.ngram1.context.phrase:
                yield u"SAME_PHRASE"


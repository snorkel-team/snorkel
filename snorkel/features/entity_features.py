from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

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

def get_ddlib_feats(context, idxs):
  """
  Minimalist port of generic mention features from ddlib
  """
  for seq_feat in _get_seq_features(context, idxs):
    yield seq_feat
  
  for window_feat in _get_window_features(context, idxs):
    yield window_feat

  if context['words'][idxs[0]][0].isupper():
      yield "STARTS_WITH_CAPTIAL"

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


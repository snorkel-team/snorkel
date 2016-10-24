import sys, os
sys.path.append(os.environ['SNORKELHOME'] + '/treedlib/treedlib')
from templates import *

try:
    from nltk.stem.porter import PorterStemmer
except ImportError:
    warnings.warn("nltk not installed- some default functionality may be absent.")

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

def get_ddlib_feats(span, idxs):
  """
  Minimalist port of generic mention features from ddlib
  """
  sent = span.parent._asdict()
  for mention_feat in _get_mention_features(span, sent, idxs):
    yield mention_feat

  for seq_feat in _get_seq_features(span, sent, idxs):
    yield seq_feat
  
  for window_feat in _get_window_features(span, sent, idxs):
    yield window_feat

  if span.get_span()[0].isupper():
      yield "STARTS_WITH_CAPTIAL"


def _get_mention_features(span, sent, idxs):
  for i in idxs:

    # The lemma of each word
    yield "WORD_LEMMA_[" + sent['lemmas'][i] + "]"

    # The part of speech of each word
    yield "WORD_POS_[" + sent['pos_tags'][i] + "]"

    # word class: convert upper-case letters to "A", lowercase letters to "a", digits to "0" and other characters to "x"
  tokens = span.get_attrib_span("words").split()
  for token in tokens:
    word_class = ""
    for ch in token:
        if ch.isupper():
            word_class = word_class + "A"
        elif ch.islower():
            word_class = word_class + "a"
        elif ch.isdigit():
            word_class = word_class + "0"
        else:
            word_class = word_class + "x"
    yield "WORD_CLASS_[" + word_class + "]"
    # Numeric normalization
    if sent['words'][i].isdigit():
        yield "WORD_NUMERIC_NORMALIZATION"
    # 2, 3 and 4-character prefixes and suffixes
    if len(token)>1: yield "WORD_PREFIX_" + token[:2]
    if len(token)>2: yield "WORD_PREFIX_" + token[:3]
    if len(token)>3: yield "WORD_PREFIX_" + token[:4]
    if len(token)>1: yield "WORD_SUFFIX_" + token[-2:]
    if len(token)>2: yield "WORD_SUFFIX_" + token[-3:]
    if len(token)>3: yield "WORD_SUFFIX_" + token[-4:]
  # 2 and 3 character n-grams
  mention = span.get_attrib_span("words")
  for i in range(len(mention) - 1):
    yield "CHARACTER_N_GRAM_[" + mention[i:i+2] + "]"
  for i in range(len(mention) - 2):
    yield "CHARACTER_N_GRAM_[" + mention[i:i+3] + "]"


def _get_seq_features(span, sent, idxs):
  if span.get_attrib_span("words") == " ".join(sent['words'][i] for i in idxs):
    stemmer = PorterStemmer()
    yield "WORD_SEQ_[" + " ".join(sent['words'][i] for i in idxs) + "]"
    yield "WORD_STEM_SEQ_[" + " ".join(stemmer.stem(sent['words'][i].lower()) for i in idxs) + "]"
    yield "LEMMA_SEQ_[" + " ".join(sent['lemmas'][i] for i in idxs) + "]"
    yield "POS_SEQ_[" + " ".join(sent['pos_tags'][i] for i in idxs) + "]"
    yield "DEP_SEQ_[" + " ".join(sent['dep_labels'][i] for i in idxs) + "]"
  else:
    tokens = span.get_attrib_span("words").split()
    stemmer = PorterStemmer()
    yield "WORD_SEQ_[" + " ".join(i for i in tokens) + "]"
    yield "WORD_STEM_SEQ_[" + " ".join(stemmer.stem(i.lower()) for i in tokens) + "]"
    yield "LEMMA_SEQ_[" + " ".join(sent['lemmas'][i] for i in idxs) + "]"
    yield "POS_SEQ_[" + " ".join(sent['pos_tags'][i] for i in idxs) + "]"
    yield "DEP_SEQ_[" + " ".join(sent['dep_labels'][i] for i in idxs) + "]"

def _get_window_features(span, sent, idxs, window=3, combinations=False, isolated=False):
    left_lemmas = []
    left_pos_tags = []
    right_lemmas = []
    right_pos_tags = []
    try:
        for i in range(1, window + 1):
            lemma = sent['lemmas'][idxs[0] - i]
            try:
                float(lemma)
                lemma = "_NUMBER"
            except ValueError:
                pass
            left_lemmas.append(lemma)
            left_pos_tags.append(sent['pos_tags'][idxs[0] - i])
    except IndexError:
        pass
    left_lemmas.reverse()
    left_pos_tags.reverse()
    try:
        for i in range(1, window + 1):
            lemma = sent['lemmas'][idxs[-1] + i]
            try:
                float(lemma)
                lemma = "_NUMBER"
            except ValueError:
                pass
            right_lemmas.append(lemma)
            right_pos_tags.append(sent['pos_tags'][idxs[-1] + i])
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

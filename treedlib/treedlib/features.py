from templates import *
import lxml.etree as et

def compile_relation_feature_generator(dictionaries=None, opts={}):
  """
  Given optional arguments, returns a generator function which accepts an xml root
  and two lists of mention indexes, and will generate relation features for this relation

  Optional args are:
    * dictionaries: should be a dictionary of lists of phrases, where the key is the dict name
    * opts: see defaults above
  """
  # TODO: put globals into opts
  #BASIC_ATTRIBS_REL = ['word', 'lemma', 'pos', 'ner', 'dep_label']
  BASIC_ATTRIBS_REL = ['lemma', 'dep_label']

  m0 = Mention(0)
  m1 = Mention(1)
  btwn = Between(m0, m1)

  dl = LengthBin(btwn, [3,4,6])
  sl = LengthBin(SeqBetween(), [5,8,14])

  # Basic relation feature templates
  templates = [

    # The full dependency path between
    [Indicator(btwn, a) for a in BASIC_ATTRIBS_REL],
    Indicator(btwn, 'dep_label,lemma'),

    # The *first element on the* path to the root: ngram lemmas along it
    Ngrams(Parents(btwn, 3), 'lemma', (1,3)),

    # The ngrams between
    #[Combinations(dl, Ngrams(btwn, a, (2,3))) for a in BASIC_ATTRIBS_REL],
    #Combinations(dl, Ngrams(btwn, 'dep_label,lemma', (2,3))),
    [Ngrams(btwn, a, (2,3)) for a in BASIC_ATTRIBS_REL],
    Ngrams(btwn, 'dep_label,lemma', (2,3)),

    # The VBs and NNs between
    #[Combinations(dl, Ngrams(Filter(btwn, 'pos', p), 'lemma', (1,3))) for p in ['VB', 'NN']],
    [Ngrams(Filter(btwn, 'pos', p), 'lemma', (1,3)) for p in ['VB', 'NN']],

    # The siblings of each mention
    [LeftNgrams(LeftSiblings(m0), a) for a in BASIC_ATTRIBS_REL],
    [LeftNgrams(LeftSiblings(m1), a) for a in BASIC_ATTRIBS_REL],
    [RightNgrams(RightSiblings(m0), a) for a in BASIC_ATTRIBS_REL],
    [RightNgrams(RightSiblings(m1), a) for a in BASIC_ATTRIBS_REL],

    # The ngrams on the *word sequence* between
    Combinations(sl, Ngrams(SeqBetween(), 'lemma', (1,3))),
    Combinations(sl, Ngrams(Filter(SeqBetween(), 'pos', 'VB'), 'lemma', (1,2))),

    # The length bin features
    sl,
    dl
  ]

  # Add dictionary features
  if dictionaries:
    for d_name, d in dictionaries.iteritems():
      templates.append(DictionaryIntersect(btwn, d_name, d))
      templates.append(DictionaryIntersect(SeqBetween(), d_name, d))

  # return generator function
  return Compile(templates).apply_relation

"""
For calibrating the bin sizes
"""
get_relation_binning_features = Compile([
  Indicator(Between(Mention(0), Mention(1)), 'word'),
  Indicator(SeqBetween(), 'word')
]).apply_relation

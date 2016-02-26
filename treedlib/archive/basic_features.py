from feature_template import *
from itertools import chain

def get_mention_templates(cid, d):
  """
  Generate the DDLib mention features as per
  http://deepdive.stanford.edu/doc/basics/gen_feats.html
  """
  return [
    
    # The set of POS tags comprising the mention
    Indicator(Mention(cid), 'pos'),

    # The set of NER tags comprising the mention
    Indicator(Mention(cid), 'ner'),

    # The set of lemmas comprising the mention
    Indicator(Mention(cid), 'lemma'),

    # The set of words comprising the mention
    Indicator(Mention(cid), 'word'),

    # TODO: Sum of the lengths of the words comprising the mention

    # Whether the first word in the mention starts with a capital letter
    RgxIndicator('^[A-Z].*$', 'word', 'STARTS_WITH_CAPITAL', Mention(cid)),

    # The lemma n-grams in a window of size 3 *of siblings* to the left and right of the mention
    # *Note* that this differs from ddlib in only considering sibling nodes
    # (which is the same if the tree is just a chain in original sequence order...)
    Indicator(Left(Mention(cid)), 'lemma'),
    Indicator(Right(Mention(cid)), 'lemma'),

    # Indicator feature of whether a keyword is part of the mention
    Indicator(Keyword(d, Mention(cid)), 'word'),

    # Indicator features of whether keywords appear in sentence
    Indicator(Keyword(d), 'word'),

    # Shortest dependency path between mention and keyword
    Indicator(Between(Mention(cid), Keyword(d)), 'lemma'),
    Indicator(Between(Mention(cid), Keyword(d)), 'dep_label')]

def get_mention_features(cid, d, root):
  return chain.from_iterable(t.apply(root) for t in get_mention_templates(cid, d))  


def get_relation_templates(cid1, cid2, d):
  """
  Generate the DDLib relation features as per
  http://deepdive.stanford.edu/doc/basics/gen_feats.html
  """
  return [
    
    # The set of POS tags comprising the mention
    Indicator(Between(Mention(cid1), Mention(cid2)), 'pos'),

    # The set of NER tags comprising the mention
    Indicator(Between(Mention(cid1), Mention(cid2)), 'ner'),

    # The set of lemmas comprising the mention
    Indicator(Between(Mention(cid1), Mention(cid2)), 'lemma'),

    # The set of words comprising the mention
    Indicator(Between(Mention(cid1), Mention(cid2)), 'word'),

    # TODO: Sum of the lengths of the words comprising the mentions

    # Whether the first word in the mentions starts with a capital letter
    RgxIndicator('^[A-Z].*$', 'word', 'STARTS_WITH_CAPITAL_1', Mention(cid1)),
    RgxIndicator('^[A-Z].*$', 'word', 'STARTS_WITH_CAPITAL_2', Mention(cid2)),

    # The n-grams of up to size 3 of the lemmas and ners of the nodes in between the mentions
    Indicator(Between(Mention(cid1), Mention(cid2), 3), 'lemma'),
    Indicator(Between(Mention(cid1), Mention(cid2), 3), 'ner'),

    # The lemma and ner n-grams in a window of size 3 *of siblings* to the left and right 
    # of the mention
    # *Note* that this differs from ddlib in only considering sibling nodes
    # (which is the same if the tree is just a chain in original sequence order...)
    Indicator(Left(Mention(cid1)), 'lemma'),
    Indicator(Left(Mention(cid2)), 'lemma'),
    Indicator(Right(Mention(cid1)), 'lemma'),
    Indicator(Right(Mention(cid2)), 'lemma'),
    Indicator(Left(Mention(cid1)), 'ner'),
    Indicator(Left(Mention(cid2)), 'ner'),
    Indicator(Right(Mention(cid1)), 'ner'),
    Indicator(Right(Mention(cid2)), 'ner'),

    # Indicator feature of whether a keyword is part of the mention
    Indicator(Keyword(d, Mention(cid1)), 'word'),
    Indicator(Keyword(d, Mention(cid2)), 'word'),

    # Indicator features of whether keywords appear in sentence
    Indicator(Keyword(d), 'word'),

    # Shortest dependency path between mention and keyword
    Indicator(Between(Mention(cid1), Keyword(d)), 'lemma'),
    Indicator(Between(Mention(cid1), Keyword(d)), 'dep_label'),
    Indicator(Between(Mention(cid2), Keyword(d)), 'lemma'),
    Indicator(Between(Mention(cid2), Keyword(d)), 'dep_label')]

def get_relation_features(cid1, cid2, d, root):
  return chain.from_iterable(t.apply(root) for t in get_relation_templates(cid1, cid2, d))  

  

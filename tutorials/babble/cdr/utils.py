import os

import bz2
from six.moves.cPickle import load

from string import punctuation

DATA_ROOT = os.environ['SNORKELHOME'] + '/tutorials/babble/cdr/data/'

def offsets_to_token(left, right, offset_array, lemmas, punc=set(punctuation)):
    token_start, token_end = None, None
    for i, c in enumerate(offset_array):
        if left >= c:
            token_start = i
        if c > right and token_end is None:
            token_end = i
            break
    token_end = len(offset_array) - 1 if token_end is None else token_end
    token_end = token_end - 1 if lemmas[token_end - 1] in punc else token_end
    return range(token_start, token_end)


class CDRTagger(object):

    def __init__(self, fname='unary_tags.pkl.bz2'):   
        with bz2.BZ2File(DATA_ROOT + fname, 'rb') as f:
            self.tag_dict = load(f)

    def tag(self, parts):
        pubmed_id, _, _, sent_start, sent_end = parts['stable_id'].split(':')
        sent_start, sent_end = int(sent_start), int(sent_end)
        tags = self.tag_dict.get(pubmed_id, {})
        for tag in tags:
            if not (sent_start <= tag[1] <= sent_end):
                continue
            offsets = [offset + sent_start for offset in parts['char_offsets']]
            toks = offsets_to_token(tag[1], tag[2], offsets, parts['lemmas'])
            for tok in toks:
                ts = tag[0].split('|')
                parts['entity_types'][tok] = ts[0]
                parts['entity_cids'][tok] = ts[1]
        return parts


class TaggerOneTagger(CDRTagger):
    
    def __init__(self, fname_tags='taggerone_unary_tags_cdr.pkl.bz2',
        fname_mesh='chem_dis_mesh_dicts.pkl.bz2'):
        with bz2.BZ2File(DATA_ROOT + fname_tags, 'rb') as f:
            self.tag_dict = load(f)
        with bz2.BZ2File(DATA_ROOT + fname_mesh, 'rb') as f:
            self.chem_mesh_dict, self.dis_mesh_dict = load(f)

    def tag(self, parts):
        parts = super(TaggerOneTagger, self).tag(parts)
        for i, word in enumerate(parts['words']):
            tag = parts['entity_types'][i]
            if len(word) > 4 and tag is None:
                wl = word.lower()
                if wl in self.dis_mesh_dict:
                    parts['entity_types'][i] = 'Disease'
                    parts['entity_cids'][i] = self.dis_mesh_dict[wl]
                elif wl in self.chem_mesh_dict:
                    parts['entity_types'][i] = 'Chemical'
                    parts['entity_cids'][i] = self.chem_mesh_dict[wl]
        return parts

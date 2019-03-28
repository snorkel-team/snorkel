import bz2
from six.moves.cPickle import load

from string import punctuation


def offsets_to_token(left, right, offset_array, lemmas, punc=set(punctuation)):
    """Find minimal token index range covering a character span
    
    For example, if a substring to be matched from an external process is "two-thirds" and 
    resides within the sentence "Earth is two-thirds water" tokenized as 
    ["Earth", "is", "two", "thirds", "water"], then a minimal range of token
    indexes covering this span is:
    
        import re
        sent, target = "Earth is two-thirds water", "two-thirds"
        words = re.compile('\w+').findall(s) # -> ['Earth', 'is', 'two', 'thirds', 'water']
        left, right = sent.index(target), sent.index(target) + len(target) - 1
        rng = offsets_to_token(left, right, [sent.index(w) for w in words], words) # -> range(2, 4)
        print([words[i] for i in rng]) # -> ["two", "thirds"]
    
    :param left: Index of first character in span
    :param right: Index of last character in span
    :param offset_array: First character index of every token in token sequence (e.g. sentence)
    :param lemmas: Lemmas of each token in sequence
    :param punc: Punctuation characters to be ignored if found at end of resulting token sequence
    :return: Minimal range of starting and ending (exclusive) token indexes that include the 
        characters at index `left` and `right`
    """
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

    def __init__(self, fname='data/unary_tags.pkl.bz2'):   
        # Load tags saved as sets of tuples for each document, e.g.:
        # {'11672959': {('Chemical|D012460', 11, 25), 'Chemical|D012460', 194, 208)}}
        with bz2.BZ2File(fname, 'rb') as f:
            self.tag_dict = load(f)

    def tag(self, parts):
        pubmed_id, _, _, sent_start, sent_end = parts['stable_id'].split(':')
        sent_start, sent_end = int(sent_start), int(sent_end)
        tags = self.tag_dict.get(pubmed_id, {})
        
        # For each tag that is a part of the provided context (e.g. sentence), assign
        # entity id and type properties for all overlapping tokens as the external
        # tagging process may have tokenized the document differently
        for tag in tags:
            if not (sent_start <= tag[1] <= sent_end):
                continue
            offsets = [offset + sent_start for offset in parts['char_offsets']]
            # Resolve the starting and ending index of characters in the tag to the 
            # tokens in the provided context that cover at least every character in
            # the tag (and possibly a few extra characters on either side of it)
            toks = offsets_to_token(tag[1], tag[2], offsets, parts['lemmas'])
            for tok in toks:
                ts = tag[0].split('|')
                parts['entity_types'][tok] = ts[0]
                parts['entity_cids'][tok] = ts[1]
        return parts


class TaggerOneTagger(CDRTagger):
    
    def __init__(self, fname_tags='data/taggerone_unary_tags_cdr.pkl.bz2',
        fname_mesh='data/chem_dis_mesh_dicts.pkl.bz2'):
        # Load tags saved as sets of tuples for each document, e.g.:
        # {'11672959': {('Chemical|D012460', 11, 25), 'Chemical|D012460', 194, 208)}}
        with bz2.BZ2File(fname_tags, 'rb') as f:
            self.tag_dict = load(f)
        # Load chemical and disease MeSH (NIH Medical Subject Heading) IDs, e.g.
        # {'leukemia, plasmacytic': 'D007952',  'erb palsy': 'D020516', ...}
        with bz2.BZ2File(fname_mesh, 'rb') as f:
            self.chem_mesh_dict, self.dis_mesh_dict = load(f)

    def tag(self, parts):
        parts = super(TaggerOneTagger, self).tag(parts)
        # Set the cid (MeSH ID in this case) and entity type properties for any words in the 
        # provided context (e.g. a sentence) that were previously tagged
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

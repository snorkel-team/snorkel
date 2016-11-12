import re
from string import punctuation
from utils import mesh_pairs_from_candidate

def get_span_splits(candidate):
    for i, s in enumerate([candidate.chemical, candidate.disease]):
        for tok in re.split(r'[\s{}]+'.format(re.escape(punctuation)), s.get_span().lower()):
            yield 'SPAN_SPLIT[{0}][{1}]'.format(i, tok), 1
            

from collections import defaultdict
from itertools import product
def get_key_ents(doc):
    chem_counts, dis_counts = defaultdict(int), defaultdict(int)
    for sent in doc.sentences:
        cur_chem, cur_dis = set(), set()
        for i, tag in enumerate(sent.ner_tags):
            t, cids = tag.split('|')[0], set(tag.split('|')[1:])
            cids.discard('-1')
            if t == 'Chemical':
                for cd in cur_dis:
                    dis_counts[cd] += 1
                cur_dis = set()
                disc = []
                for cc in cur_chem:
                    if cc not in cids:
                        chem_counts[cc] += 1
                        disc.append(cc)
                for d in disc:
                    cur_chem.discard(d)
                for cid in cids:
                    cur_chem.add(cid)                        
            elif t == 'Disease':
                for cc in cur_chem:
                    chem_counts[cc] += 1
                cur_chem = set()
                disc = []
                for cd in cur_dis:
                    if cd not in cids:
                        dis_counts[cd] += 1
                        disc.append(cd)
                for d in disc:
                    cur_dis.discard(d)
                for cid in cids:
                    cur_dis.add(cid)
            else:
                for cd in cur_dis:
                    dis_counts[cd] += 1
                cur_dis = set()
                for cc in cur_chem:
                    chem_counts[cc] += 1
                cur_chem = set()
    if len(chem_counts) == 0:
        max_chem = set()
    else:
        m_chem = max(chem_counts.values())
        max_chem = set([chem for chem, count in chem_counts.iteritems() if count == m_chem])
    if len(dis_counts) == 0:
        max_dis = set()
    else:
        m_dis = max(dis_counts.values())
        max_dis = set([dis for dis, count in dis_counts.iteritems() if count == m_dis])

    
    return max_chem, max_dis

def get_is_key(candidate):
    doc = candidate[0].parent.document
    key_chem, key_dis = get_key_ents(doc)
    chem_flag, dis_flag = False, False
    pubmed_id, pairs = mesh_pairs_from_candidate(candidate)
    pairs = list(pairs)
    for chem, dis in pairs:
        if not chem_flag and (chem in key_chem):
            chem_flag = True
            yield 'KEY_CHEMICAL', 1
        if not dis_flag and (dis in key_dis):
            dis_flag = True
            yield 'KEY_DISEASE', 1
        if chem_flag and dis_flag:
            yield 'KEY_CHEMICAL_AND_DISEASE', 1
            break
    chem_flag, dis_flag = False, False
    for chem, dis in pairs:
        if not chem_flag and any([chem in tag for tag in doc.sentences[0].ner_tags]):
            chem_flag = True
            yield 'TITLE_CHEMICAL', 1
        if not dis_flag and any([dis in tag for tag in doc.sentences[0].ner_tags]):
            dis_flag = True
            yield 'TITLE_DISEASE', 1
        if chem_flag and dis_flag:
            yield 'TITLE_CHEMICAL_AND_DISEASE', 1
            break

import os, sys
sys.path.append(os.path.join(os.environ['SNORKELHOME'], 'treedlib'))
from treedlib import compile_relation_feature_generator
from tree_structs import corenlp_to_xmltree
from snorkel.utils import get_as_dict

def get_title_span_feats(candidate, stopwords=None):
    title_sent = candidate[0].parent.document.sentences[0]
    _, pairs = mesh_pairs_from_candidate(candidate)
    for chem, dis in pairs:
        s1_idxs, s2_idxs = [], []
        for i, tag in enumerate(title_sent.ner_tags):
            if (not chem in tag) and (not dis in tag):
                continue
            if (chem in tag) and (len(s1_idxs) == 0 or s1_idxs[-1] == (i-1)):
                s1_idxs.append(i)
                continue
            if (dis in tag) and (len(s2_idxs) == 0 or s2_idxs[-1] == (i-1)):
                s2_idxs.append(i)
                continue
        if len(s1_idxs) > 0 and len(s2_idxs) > 0:
            break 
    if len(s1_idxs) > 0 and len(s2_idxs) > 0:
        get_tdl_feats = compile_relation_feature_generator()
        xmltree       = corenlp_to_xmltree(get_as_dict(title_sent))
        for f in get_tdl_feats(xmltree.root, s1_idxs, s2_idxs, stopwords=stopwords):
            yield 'TDL_' + f, 1
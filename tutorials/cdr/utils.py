import cPickle
import lxml.etree as et
import re
import string

from itertools import product
from snorkel.lf_helpers import get_tagged_text, get_text_between
from snorkel.parser import SentenceParser, Sentence


def mesh_pairs_from_candidate(candidate):
    pubmed_id = candidate[0].parent.document.stable_id.split(':')[0]
    chem_tokens = range(candidate[0].get_word_start(), candidate[0].get_word_end() + 1)
    chem_mesh = list(set(sum([candidate[0].parent.ner_tags[t].split('|')[1:] for t in chem_tokens], [])))
    dis_tokens = range(candidate[1].get_word_start(), candidate[1].get_word_end() + 1)
    dis_mesh = list(set(sum([candidate[1].parent.ner_tags[t].split('|')[1:] for t in dis_tokens], [])))
    return pubmed_id, product(chem_mesh, dis_mesh)


def offsets_to_token(left, right, offset_array, lemmas, punc=set(string.punctuation)):
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


class TaggerOneSentenceParser(SentenceParser):
    
    tag_dict = cPickle.load(open('data/taggerone_unary_tags_cdr.pkl', 'rb'))
    chem_mesh_dict, dis_mesh_dict = cPickle.load(open('data/chem_dis_mesh_dicts.pkl', 'rb'))
    
    def parse(self, doc, text):
        """Parse a raw document as a string into a list of sentences, subbing in TaggerOne tags"""
        for parts in self.corenlp_handler.parse(doc, text):
            pubmed_id, _, _, sent_start, sent_end = parts['stable_id'].split(':')
            sent_start, sent_end = int(sent_start), int(sent_end)
            tags = self.tag_dict.get(pubmed_id, {})
            for tag in tags:
                if not (sent_start <= tag[1] <= sent_end):
                    continue
                offsets = [offset + sent_start for offset in parts['char_offsets']]
                toks = offsets_to_token(tag[1], tag[2], offsets, parts['lemmas'])
                for tok in toks:
                    parts['ner_tags'][tok] = tag[0]
                    
            for i, word in enumerate(parts['words']):
                tag = parts['ner_tags'][i]
                if len(word) > 4 and not (tag.startswith('Chemical') or tag.startswith('Disease')):
                    wl = word.lower()
                    if wl in self.dis_mesh_dict:
                        parts['ner_tags'][i] = 'Disease|' + self.dis_mesh_dict[wl]
                    elif wl in self.chem_mesh_dict:
                        parts['ner_tags'][i] = 'Chemical|' + self.chem_mesh_dict[wl]
                        
            yield Sentence(**parts)


class CDRSentenceParser(SentenceParser):
    
    tag_dict = cPickle.load(open('data/unary_tags.pkl', 'rb'))
    
    def parse(self, doc, text):
        possible_fixed = set()
        for parts in self.corenlp_handler.parse(doc, text):
            pubmed_id, _, _, sent_start, sent_end = parts['stable_id'].split(':')
            sent_start, sent_end = int(sent_start), int(sent_end)
            tags = self.tag_dict.get(pubmed_id, {})
            for tag in tags:
                if not (sent_start <= tag[1] <= sent_end):
                    continue
                offsets = [offset + sent_start for offset in parts['char_offsets']]
                toks = offsets_to_token(tag[1], tag[2], offsets, parts['lemmas'])
                for tok in toks:
                    parts['ner_tags'][tok] = tag[0]                        
            yield Sentence(**parts)

            
def gen_LF_text_btw(c, text, sign=1):
    return sign if text in get_text_between(c) else 0

def gen_LF_span(c, text, span=0, sign=1):
    return sign if text in c[span].get_span().lower() else 0   

def gen_LF_regex(c, pattern, sign):
    return sign if re.search(pattern, get_tagged_text(c), flags=re.I) else 0
    
def gen_LF_regex_AB(c, pattern, sign):
    return sign if re.search(r'{{A}}' + pattern + r'{{B}}', get_tagged_text(c), flags=re.I) else 0

def gen_LF_regex_BA(c, pattern, sign):
    return sign if re.search(r'{{B}}' + pattern + r'{{A}}', get_tagged_text(c), flags=re.I) else 0
    
def gen_LF_regex_A(c, pattern, sign):
    return sign if re.search(pattern + r'{{A}}.*{{B}}', get_tagged_text(c), flags=re.I) else 0
    
def gen_LF_regex_B(c, pattern, sign):
    return sign if re.search(pattern + r'{{B}}.*{{A}}', get_tagged_text(c), flags=re.I) else 0
    
def ltp(x):
    return '(' + '|'.join(x) + ')'

###########################################################################################################
from snorkel.learning import FMCT
from snorkel.learning.utils import score, test_scores
from collections import defaultdict
from pandas import DataFrame
import numpy as np
from itertools import product
from cdr_feats import get_key_ents

def get_doc_from_id(doc_id, corpus):
    for d in corpus:
        if d.name == doc_id:
            return d
    print "Couldn't find candidate with doc id {0}".format(doc_id)
    return None

def get_important_chems(doc_id, corpus):
    doc = get_doc_from_id(doc_id, corpus)
    if doc is None:
        return []
    title_sent = doc.sentences[0]
    key_chems = set([
        tag.split('|')[1] for tag in title_sent.ner_tags if tag.startswith('Chemical')
    ])
    if len(key_chems) == 0:
        key_chems, _ = get_key_ents(doc)
    return list(key_chems)
    
def get_all_diseases(doc_id, corpus):
    doc = get_doc_from_id(doc_id, corpus)
    if doc is None:
        return []
    return list(set([            
        tag.split('|')[1] for sent in doc.sentences for tag in sent.ner_tags if tag.startswith('Disease')
    ]))



class CandidateHolder(object):
    def __init__(self, p=0):
        self.p = p
        self.candidates = set()

        
chem_filter_ratio = cPickle.load(open('data/train_dev_chem_filter.pkl', 'rb'))
dis_filter_ratio = cPickle.load(open('data/train_dev_dis_filter.pkl', 'rb'))


def cdr_doc_score(test_marginals, doc_relation_dict, gold_candidate_set, corpus, b=None, filt_p=5):
    print "Scoring\t"
    max_f1 = 0
    p_path, r_path, f_path = [], [], []
    bs = [0.4, 0.5, 0.6] if b is None else [b]
    for b in bs:

        # Group test candidates by doc
        ent_dict = defaultdict(dict)
        for i, candidate in enumerate(gold_candidate_set):
            pubmed_id, pairs = mesh_pairs_from_candidate(candidate)
            # Record the maximum probability for the candidate found in the document
            for c, d in pairs:
                pair = (c, d)
                if '|' in c or '|' in d or c == '-1' or d == '-1':
                    continue
                holder = ent_dict[pubmed_id].get(pair, CandidateHolder())
                holder.p = max(holder.p, test_marginals[i])
                holder.candidates.add(candidate)
                ent_dict[pubmed_id][pair] = holder

        ##################################################################
        # Recall increasing heuristic
        rih_docs = []
        for doc_id in doc_relation_dict:
            if doc_id not in ent_dict:
                rih_docs.append(doc_id)
            else:
                for holder in ent_dict[doc_id].values():
                    if holder.p > b:
                        break
                else:
                    rih_docs.append(doc_id)

        n_up = 0
        for doc_id in rih_docs:
            chem_ids = get_important_chems(doc_id, corpus)
            dis_ids = get_all_diseases(doc_id, corpus)
            for c, d in product(chem_ids, dis_ids):
                n_up += 1
                pair = (c, d)
                holder = ent_dict[doc_id].get(pair, CandidateHolder())
                holder.p = 1.0
                ent_dict[doc_id][pair] = holder 
        ##################################################################

        ##################################################################
        # Filter
        chem_filter = set(k for k,v in chem_filter_ratio.items() if v >= filt_p)
        dis_filter = set(k for k,v in dis_filter_ratio.items() if v >= filt_p)
        n_filt = 0
        for doc_id in ent_dict:
            for pair in ent_dict[doc_id]:
                if pair[0] in chem_filter or pair[1] in dis_filter:
                    ent_dict[doc_id][pair].p = 0
                    n_filt += 1
        ##################################################################


        predict = []
        test_labels = []
        tp = set()
        fp = set()
        tn = set()
        fn = set()
        for pubmed_id, relation_entities in ent_dict.iteritems():
            # Iterate over all relation entities in the doc
            for rel, holder in relation_entities.iteritems():
                marginal = holder.p
                predict.append(1 if marginal > b else (-1 if marginal < b else 0))
                if rel in doc_relation_dict[pubmed_id]:
                    test_labels.append(1)
                    if marginal > b:
                        tp = tp.union(holder.candidates)
                    else:
                        fn = fn.union(holder.candidates)
                else:
                    test_labels.append(-1)
                    if marginal > b:
                        fp = fp.union(holder.candidates)
                    else:
                        tn = tn.union(holder.candidates)

        gold_set = set((doc_key, rel) for doc_key, rels in doc_relation_dict.iteritems() for rel in rels)
        cand_set = set((doc_key, rel_key) for doc_key, rels in ent_dict.iteritems() for rel_key in rels.keys())
        extra_fn = len(gold_set.difference(cand_set))



        # Print diagnostics chart and return error analysis candidate sets
        predict, test_labels = np.ravel(predict), np.ravel(test_labels)
        _, _, _, m_tp, m_fp, m_tn, m_fn, m_n_t = test_scores(predict, test_labels, verbose=False)
        mm_fn = m_fn + extra_fn
        prec = m_tp / float(m_tp + m_fp)
        rec  = m_tp / float(m_tp + mm_fn)
        f1 = 2.0 * (prec * rec) / (prec + rec)
        pos_acc = m_tp/float(m_tp+mm_fn)
        neg_acc = m_tn/float(m_tn+m_fp)

        if f1 >= max_f1:
            max_tp, max_fp, max_tn, max_fn, max_prec, max_rec, max_f1 = tp, fp, tn, fn, prec, rec, f1
            max_pos_acc, max_neg_acc, max_m_tp, max_m_fp, max_m_tn, max_mm_fn = pos_acc, neg_acc, m_tp, m_fp, m_tn, mm_fn
            b_max, filt_p_max = b, filt_p

    print "========================================================="
    print "Recall-corrected Noise-aware Model @ b={0} and filter={1}".format(b_max, filt_p_max)
    print "========================================================="
    print "Pos. class accuracy: {:.3}".format(max_pos_acc)
    print "Neg. class accuracy: {:.3}".format(max_neg_acc)
    print "Corpus Precision {:.3}".format(max_prec)
    print "Corpus Recall    {:.3}".format(max_rec)
    print "Corpus F1        {:.3}".format(max_f1)
    print "----------------------------------------"
    print "TP: {} | FP: {} | TN: {} | FN: {}".format(max_m_tp, max_m_fp, max_m_tn, max_mm_fn)
    print "========================================\n"

    return max_tp, max_fp, max_tn, max_fn, max_prec, max_rec, max_f1


class CDRFMCT(FMCT):
    def score(self, X_test, X_test_raw, doc_relation_dict, gold_candidate_set, corpus, b=None, filt_p=5):
        print "Predicting\t",
        test_marginals = self.marginals(X_test, X_test_raw)
        return cdr_doc_score(test_marginals, doc_relation_dict, gold_candidate_set, corpus, b, filt_p)
    
    
from snorkel.learning.utils import RandomSearch, ListParameter, RangeParameter

class CDRRandomSearch(RandomSearch):
    def fit(self, X_validation, X_validation_raw, doc_relation_dict, gold_candidate_set, corpus, b=0.5, **model_hyperparams):
        # Iterate over the param values
        self.run_stats   = []
        param_opts  = np.zeros(len(self.param_names))
        f1_opt      = -1.0
        for k, param_vals in enumerate(self.search_space()):

            # Set the new hyperparam configuration to test
            for pn, pv in zip(self.param_names, param_vals):
                model_hyperparams[pn] = pv
            print "=" * 60
            print "{%d}: Testing %s" % (k+1, ', '.join(["{0} = {1}".format(pn,pv) for pn,pv in zip(self.param_names, param_vals)]))
            print "=" * 60

            # Train the model
            self.model.train(self.X, self.training_marginals, **model_hyperparams)

            # Test the model
            tp, fp, tn, fn, p, r, f1 = self.model.score(X_validation, X_validation_raw, doc_relation_dict, gold_candidate_set, corpus, b)
            self.run_stats.append(list(param_vals) + [p, r, f1])
            if f1 > f1_opt:
                model_opt  = self.model.fmct
                param_opts = param_vals
                f1_opt     = f1

        # Set optimal parameter in the learner model
        self.model.fmct = model_opt

        # Return DataFrame of scores
        self.results = DataFrame.from_records(self.run_stats, columns=self.param_names + ['Prec.', 'Rec.', 'F1']).sort('F1', ascending=False)
        return self.results
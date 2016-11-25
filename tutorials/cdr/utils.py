import cPickle

from collections import defaultdict
from itertools import product
from pandas import DataFrame
from snorkel.learning import FMCT
from snorkel.learning.utils import print_scores, RandomSearch, Scorer
from string import punctuation


def pubmed_id_from_candidate(candidate):
    return candidate[0].parent.document.stable_id.split(':')[0]


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
    
    tag_dict = cPickle.load(open('data/unary_tags.pkl', 'rb'))

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
    
    tag_dict = cPickle.load(open('data/taggerone_unary_tags_cdr.pkl', 'rb'))
    chem_mesh_dict, dis_mesh_dict = cPickle.load(open('data/chem_dis_mesh_dicts.pkl', 'rb'))

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


###########################################################################################################


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
        chem_counts = defaultdict(int)
        for sent in doc.sentences:
            cur_chem = None
            for i, (tag, cid) in enumerate(zip(sent.entity_types, entity_cids)):
                if tag == 'Chemical':
                    if cid != cur_chem and cur_chem is not None:
                        chem_counts[cur_chem] += 1
                    cur_chem = cid
                else:
                    if cur_chem is not None:
                        chem_counts[cur_chem] += 1
                    cur_chem = None
        if len(chem_counts) == 0:
            key_chems = set()
        else:
            m_chem = max(chem_counts.values())
            key_chems = set([chem for chem, count in chem_counts.iteritems() if count == m_chem])
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


class CDRScorer(Scorer):
    def score(self, test_marginals, train_marginals=None, b=0.5, set_unlabeled_as_neg=True, display=True):
        max_f1 = 0
        bs = [0.4, 0.5, 0.6] if b is None else [b]
        for b in bs:
            # Group test candidates by doc
            ent_dict = defaultdict(dict)
            for i, candidate in enumerate(gold_candidate_set):
                pubmed_id, pair = pubmed_id_from_candidate(candidate), candidate.get_cids()
                # Record the maximum probability for the candidate found in the document
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
            for doc_id in rih_docs:
                chem_ids = get_important_chems(doc_id, corpus)
                dis_ids = get_all_diseases(doc_id, corpus)
                for c, d in product(chem_ids, dis_ids):
                    pair = (c, d)
                    holder = ent_dict[doc_id].get(pair, CandidateHolder())
                    holder.p = 1.0
                    ent_dict[doc_id][pair] = holder

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
            # Score after RIH and filter
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
            # Report best score
            gold_set = set((doc_key, rel) for doc_key, rels in doc_relation_dict.iteritems() for rel in rels)
            cand_set = set((doc_key, rel_key) for doc_key, rels in ent_dict.iteritems() for rel_key in rels.keys())
            extra_fn = len(gold_set.difference(cand_set))
            prec = m_tp / float(m_tp + m_fp)
            rec  = m_tp / float(m_tp + mm_fn)
            f1 = 2.0 * (prec * rec) / (prec + rec)
            if f1 > max_f1:
                max_tp, max_fp, max_tn, max_fn, max_extra_fn = tp, fp, tn, fn, extra_fn
                max_f1 = f1
        # Calculate scores unadjusted for TPs not in our candidate set
        print_scores(len(max_tp), len(max_p), len(max_tn), len(max_fn), title="Scores (Un-adjusted)")
        # If a gold candidate set is provided, also calculate recall-adjusted scores
        print "\n"
        print_scores(len(max_tp), len(max_fp), len(max_tn), len(max_fn)+max_extra_fn,title="Corpus Recall-adjusted Scores")
        return tp, fp, tn, fn


class CDRFMCT(FMCT):
    def score(self, X_test, X_test_raw, doc_relation_dict, gold_candidate_set, corpus, b=None, filt_p=5):
        test_marginals = self.marginals(X_test, X_test_raw)
        return cdr_doc_score(test_marginals, doc_relation_dict, gold_candidate_set, corpus, b, filt_p)


class CDRRandomSearch(RandomSearch):
    def fit(self, X_validation, X_validation_raw, doc_relation_dict, gold_candidate_set, corpus, b=0.5, **model_hyperparams):
        # Iterate over the param values
        self.run_stats   = []
        param_opts  = None
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

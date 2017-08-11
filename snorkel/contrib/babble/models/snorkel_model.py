# Python
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import csv
import cPickle
import bz2
from pprint import pprint
from scipy.sparse import coo_matrix

# Snorkel
from snorkel.models import Document, Sentence, candidate_subclass
from snorkel.parser import CorpusParser, TSVDocPreprocessor, XMLMultiDocPreprocessor
from snorkel.parser.spacy_parser import Spacy
from snorkel.candidates import Ngrams, CandidateExtractor
from snorkel.matchers import PersonMatcher
from snorkel.annotations import (FeatureAnnotator, LabelAnnotator, 
    save_marginals, load_marginals, load_gold_labels)
from snorkel.learning import GenerativeModel, SparseLogisticRegression, MajorityVoter
from snorkel.learning import RandomSearch, ListParameter, RangeParameter
from snorkel.learning.utils import MentionScorer, training_set_summary_stats
from snorkel.learning.structure import DependencySelector
from snorkel.learning.disc_models.rnn import reRNN


TRAIN = 0
DEV = 1
TEST = 2

class SnorkelModel(object):
    """
    A class for running a complete Snorkel pipeline
    """
    def __init__(self, session, candidate_class, config):
        self.session = session
        self.candidate_class = candidate_class
        self.config = config

        if config['seed']:
            np.random.seed(config['seed'])

        self.lfs = None
        self.labeler = None
        self.featurizer = None

    def get_candidates(split):
        self.session.query(self.candidate_class).filter(
                self.candidate_class.split == split).order_by(self.candidate_class.id).all()

    def parse(self, doc_preprocessor, parser=Spacy(), fn=None, clear=True):
        corpus_parser = CorpusParser(parser=parser, fn=fn)
        corpus_parser.apply(doc_preprocessor, count=doc_preprocessor.max_docs, 
                            parallelism=self.config['parallelism'], clear=clear)
        if self.config['verbose']:
            print("Documents: {}".format(self.session.query(Document).count()))
            print("Sentences: {}".format(self.session.query(Sentence).count()))        

    def extract(self, cand_extractor, sents, split, clear=True):
        cand_extractor.apply(sents, split=split, parallelism=self.config['parallelism'], clear=clear)
        nCandidates = self.session.query(self.candidate_class).filter(self.candidate_class.split == split).count()
        if self.config['verbose']:
            print("Candidates [Split {}]: {}".format(split, nCandidates))

    def load_gold(self):
        raise NotImplementedError

    def featurize(self, featurizer, split):
        if split == TRAIN:
            F = featurizer.apply(split=split, parallelism=self.config['parallelism'])
        else:
            F = featurizer.apply_existing(split=split, parallelism=self.config['parallelism'])
        nCandidates, nFeatures = F.shape
        if self.config['verbose']:
            print("\nFeaturized split {}: ({},{}) sparse (nnz = {})".format(split, nCandidates, nFeatures, F.nnz))
        return F

    def label(self, labeler, split):
        if split == TRAIN:
            L = labeler.apply(split=split, parallelism=self.config['parallelism'])
        else:
            L = labeler.apply_existing(split=split, parallelism=self.config['parallelism'])
        return L
    
    def supervise(self, config=None):
        if config:
            self.config = config

        if not self.labeler:
            if self.lfs:
                self.labeler = LabelAnnotator(lfs=self.lfs)
            else:
                raise Exception("Cannot load label matrix without having LF list.")
        L_train = self.labeler.load_matrix(self.session, split=TRAIN)

        if self.config['traditional']:
            # Do traditional supervision with hard labels
            L_gold_train = load_gold_labels(self.session, annotator_name='gold', split=TRAIN)
            train_marginals = np.array(L_gold_train.todense()).reshape((L_gold_train.shape[0],))
            train_marginals[train_marginals==-1] = 0
        else:
            if self.config['majority_vote']:
                self.gen_model = MajorityVoter()
            else:
                if self.config['model_dep']:
                    ds = DependencySelector()
                    deps = ds.select(L_train, threshold=self.config['threshold'])
                    if self.config['verbose']:
                        self.display_dependencies(deps)
                else:
                    deps = ()
            
                self.gen_model = GenerativeModel(lf_propensity=True)
                
                decay = (self.config['decay'] if self.config['decay'] else 
                    0.001 * (1.0 /self.config['epochs']))
                step_size = (self.config['step_size'] if self.config['step_size'] else 
                    0.1/L_train.shape[0])
                self.gen_model.train(
                    L_train, 
                    deps=deps, 
                    epochs=self.config['epochs'],
                    decay=decay,
                    step_size=step_size,
                    reg_param=self.config['reg_param'])

            train_marginals = self.gen_model.marginals(L_train)
                
            if self.config['majority_vote']:
                self.lf_stats = None
            else:
                if self.config['verbose']:
                    if self.config['empirical_from_train']:
                        L = self.labeler.load_matrix(self.session, split=TRAIN)
                        L_gold = load_gold_labels(self.session, annotator_name='gold', split=TRAIN)
                    else:
                        L = self.labeler.load_matrix(self.session, split=DEV)
                        L_gold = load_gold_labels(self.session, annotator_name='gold', split=DEV)
                    self.lf_stats = L.lf_stats(self.session, L_gold, self.gen_model.weights.lf_accuracy())
                    if self.config['display_correlation']:
                        self.display_accuracy_correlation()
            
        save_marginals(self.session, L_train, train_marginals)

        if self.config['verbose']:
            if self.config['display_marginals']:
                # Display marginals
                plt.hist(train_marginals, bins=20)
                plt.show()

    def classify(self, config=None):
        if config:
            self.config = config

        if self.config['seed']:
            np.random.seed(self.config['seed'])

        train_marginals = load_marginals(self.session, split=TRAIN)

        if DEV in self.config['splits']:
            L_gold_dev = load_gold_labels(self.session, annotator_name='gold', split=DEV)
        if TEST in self.config['splits']:
            L_gold_test = load_gold_labels(self.session, annotator_name='gold', split=TEST)

        if self.config['disc_model']=='logreg':
            disc_model = SparseLogisticRegression()
            self.model = disc_model

            if not self.featurizer:
                self.featurizer = FeatureAnnotator()
            if TRAIN in self.config['splits']:
                F_train =  self.featurizer.load_matrix(self.session, split=TRAIN)
            if DEV in self.config['splits']:
                F_dev =  self.featurizer.load_matrix(self.session, split=DEV)
            if TEST in self.config['splits']:
                F_test =  self.featurizer.load_matrix(self.session, split=TEST)

            if self.config['traditional']:
                train_size = self.config['traditional']
                F_train = F_train[:train_size, :]
                train_marginals = train_marginals[:train_size]
                print("Using {0} hard-labeled examples for supervision\n".format(train_marginals.shape[0]))

            if self.config['num_search'] > 1:
                lr_min, lr_max = min(self.config['lr']), max(self.config['lr'])
                l1_min, l1_max = min(self.config['l1_penalty']), max(self.config['l1_penalty'])
                l2_min, l2_max = min(self.config['l2_penalty']), max(self.config['l2_penalty'])
                lr_param = RangeParameter('lr', lr_min, lr_max, step=1, log_base=10)
                l1_param  = RangeParameter('l1_penalty', l1_min, l1_max, step=1, log_base=10)
                l2_param  = RangeParameter('l2_penalty', l2_min, l2_max, step=1, log_base=10)
            
                searcher = RandomSearch(self.session, disc_model, 
                                        F_train, train_marginals, 
                                        [lr_param, l1_param, l2_param], 
                                        n=self.config['num_search'])

                print("\nRandom Search:")
                search_stats = searcher.fit(F_dev, L_gold_dev, 
                                            n_epochs=self.config['num_epochs'], 
                                            rebalance=self.config['rebalance'],
                                            print_freq=self.config['print_freq'],
                                            seed=self.config['seed'])

                if self.config['verbose']:
                    print(search_stats)
                
                disc_model = searcher.model
                    
            else:
                lr = self.config['lr'] if len(self.config['lr'])==1 else 1e-2
                l1_penalty = self.config['l1_penalty'] if len(self.config['l1_penalty'])==1 else 1e-3
                l2_penalty = self.config['l2_penalty'] if len(self.config['l2_penalty'])==1 else 1e-5
                disc_model.train(F_train, train_marginals, 
                                 lr=lr, 
                                 l1_penalty=l1_penalty, 
                                 l2_penalty=l2_penalty,
                                 n_epochs=self.config['num_epochs'], 
                                 rebalance=self.config['rebalance'],
                                 seed=self.config['seed'])
            
            if DEV in self.config['splits']:
                print("\nDev:")
                TP, FP, TN, FN = disc_model.score(self.session, F_dev, L_gold_dev, train_marginals=train_marginals, b=self.config['b'])
            
            if TEST in self.config['splits']:
                print("\nTest:")
                TP, FP, TN, FN = disc_model.score(self.session, F_test, L_gold_test, train_marginals=train_marginals, b=self.config['b'])

        elif self.config['disc_model'] == 'lstm':
            print("Warning: LSTM params are currently hardcoded!")
            train_cands = self.get_candidates(TRAIN)
            dev_cands = self.get_candidates(DEV)
            test_cands = self.get_candidates(TEST)
            
            train_kwargs = {
                'lr':         0.01,
                'dim':        50,
                'n_epochs':   10,
                'dropout':    0.25,
                'print_freq': 1,
                'max_sentence_length': 100
            }
            lstm = reRNN(seed=1701, n_threads=None)
            lstm.train(train_cands, train_marginals, X_dev=dev_cands, Y_dev=L_gold_dev, **train_kwargs)

            if TEST in self.config['splits']:
                print("\nTest:")
                p, r, f1 = lstm.score(test_cands, L_gold_test)
                print("Prec: {0:.3f}, Recall: {1:.3f}, F1 Score: {2:.3f}".format(p, r, f1))

                tp, fp, tn, fn = lstm.error_analysis(self.session, test_cands, L_gold_test)

            self.disc_model = lstm

        else:
            raise NotImplementedError

    def display_accuracy_correlation(self):
        """Displays ..."""
        empirical = self.lf_stats['Empirical Acc.'].get_values()
        learned = self.lf_stats['Learned Acc.'].get_values()
        conflict = self.lf_stats['Conflicts'].get_values()
        N = len(learned)
        colors = np.random.rand(N)
        area = np.pi * (30 * conflict)**2  # 0 to 30 point radii
        plt.scatter(empirical, learned, s=area, c=colors, alpha=0.5)
        plt.xlabel('empirical')
        plt.ylabel('learned')
        plt.show()

    def display_dependencies(self, deps_encoded):
        """Displays ..."""
        dep_names = {
            0: 'DEP_SIMILAR',
            1: 'DEP_FIXING',
            2: 'DEP_REINFORCING',
            3: 'DEP_EXCLUSIVE',
        }
        if not self.lfs:
            self.generate_lfs()
            print("Running generate_lfs() first...")   
        lf_names = {i:lf.__name__ for i, lf in enumerate(self.lfs)}
        deps_decoded = []
        for dep in deps_encoded:
            (lf1, lf2, d) = dep
            deps_decoded.append((lf_names[lf1], lf_names[lf2], dep_names[d]))
        for dep in sorted(deps_decoded):
            (lf1, lf2, d) = dep
            print('{:16}: ({}, {})'.format(d, lf1, lf2))

            # lfs = sorted([lf1, lf2])
            # deps_decoded.append((lf_names[lfs[0]], lf_names[lfs[1]], dep_names[d]))
        # for dep in sorted(list(set(deps_decoded))):
        #     (lf1, lf2, d) = dep
        #     print('{:16}: ({}, {})'.format(d, lf1, lf2))
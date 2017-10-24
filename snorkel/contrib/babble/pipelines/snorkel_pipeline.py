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
    save_marginals, load_marginals, load_gold_labels, load_label_matrix)
from snorkel.learning import GenerativeModel, MajorityVoter
from snorkel.learning.structure import DependencySelector
from snorkel.learning import reRNN

# Pipelines
from utils import STAGES, PrintTimer, train_model, score_marginals, final_report

TRAIN = 0
DEV = 1
TEST = 2

class SnorkelPipeline(object):
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


    def run(self):
        def is_valid_stage(stage_id):
            return self.config['start_at'] <= stage_id < self.config['end_at']
                    
        if self.config['start_at'] is None or self.config['end_at'] is None:
            raise Exception("At least one of 'start_at' or 'end_at' is not defined.")

        if self.config['debug']:
            print("NOTE: --debug=True: modifying parameters...")
            self.config['max_docs'] = 100
            self.config['gen_model_search_space'] = 2
            self.config['disc_model_search_space'] = 2
            self.config['gen_params_default']['epochs'] = 25
            self.config['disc_params_default']['n_epochs'] = 5

        result = None
        for stage in ['parse', 'extract', 'load_gold', 'collect', 'label', 
                      'supervise', 'classify']:
            stage_id = getattr(STAGES, stage.upper())
            if is_valid_stage(stage_id):
                with PrintTimer('[{}] {}...'.format(stage_id, stage.capitalize())):
                    result = getattr(self, stage)()

        return result


    def parse(self, doc_preprocessor, parser=Spacy(), fn=None, clear=True):
        corpus_parser = CorpusParser(parser=parser, fn=fn)
        corpus_parser.apply(doc_preprocessor, count=doc_preprocessor.max_docs, 
                            parallelism=self.config['parallelism'], clear=clear)
        if self.config['verbose']:
            print("Documents: {}".format(self.session.query(Document).count()))
            print("Sentences: {}".format(self.session.query(Sentence).count()))        


    def extract(self, cand_extractor, sents, split, clear=True):
        cand_extractor.apply(sents, split=split, parallelism=self.config['parallelism'], clear=clear)
        if self.config['verbose']:
            num_candidates = self.session.query(self.candidate_class).filter(self.candidate_class.split == split).count()
            print("Candidates [Split {}]: {}".format(split, num_candidates))


    def load_gold(self):
        raise NotImplementedError


    def featurize(self, featurizer, split):
        if split == TRAIN:
            F = featurizer.apply(split=split, parallelism=self.config['parallelism'])
        else:
            F = featurizer.apply_existing(split=split, parallelism=self.config['parallelism'])
        num_candidates, num_features = F.shape
        if self.config['verbose']:
            print("\nFeaturized split {}: ({},{}) sparse (nnz = {})".format(split, num_candidates, num_features, F.nnz))
        return F


    def collect(self):
        raise NotImplementedError


    def label(self, labeler, split):
        if split == TRAIN:
            L = labeler.apply(split=split, parallelism=self.config['parallelism'])
        else:
            L = labeler.apply_existing(split=split, parallelism=self.config['parallelism'])
        return L
    

    def supervise(self, config=None):
        """Calculate and save L_train and train_marginals.

        traditional: use known 0 or 1 labels from gold data
        majority_vote: use majority vote on LF outputs
        [default]: use generative model
            learn_deps: learn the dependencies of the generative model
        """
        if config:
            self.config = config

        if self.config['supervision'] == 'traditional':
            L_gold_train = load_gold_labels(self.session, annotator_name='gold', split=TRAIN)
            L_train, train_marginals = self.traditional_supervision(L_gold_train)
            gen_model = None
            if self.config['display_marginals'] and not self.config['no_plots']:
                plt.hist(train_marginals, bins=20)
                plt.show()
        else:
            if not getattr(self, 'L_train', None):
                self.L_train = load_label_matrix(self.session, split=TRAIN)
            L_train = self.L_train
            assert L_train.nnz > 0
            if self.config['verbose']:
                print("Using L_train: {0}".format(L_train.__repr__()))

            # Load DEV and TEST labels and gold labels
            if DEV in self.config['splits']:
                L_dev = load_label_matrix(self.session, split=DEV)
                assert L_dev.nnz > 0
                if self.config['verbose']:
                    print("Using L_dev: {0}".format(L_dev.__repr__()))
                L_gold_dev = load_gold_labels(self.session, annotator_name='gold', split=DEV)
                assert L_gold_dev.nonzero()[0].shape[0] > 0
            if TEST in self.config['splits']:
                L_test = load_label_matrix(self.session, split=TEST)
                assert L_test.nnz > 0
                if self.config['verbose']:
                    print("Using L_test: {0}".format(L_test.__repr__()))
                L_gold_test = load_gold_labels(self.session, annotator_name='gold', split=TEST)
                assert L_gold_test.nonzero()[0].shape[0] > 0

            if self.config['supervision'] == 'majority_vote':
                gen_model = MajorityVoter()
                train_marginals = gen_model.marginals(L_train)

            elif self.config['supervision'] == 'generative':

                # Learn dependencies
                if self.config['learn_deps']:
                    ds = DependencySelector()
                    deps = ds.select(L_train, threshold=self.config['threshold'])
                    if self.config['verbose']:
                        self.display_dependencies(deps)
                else:
                    deps = ()

                # Pass in the dependencies via default params
                gen_params_default = self.config['gen_params_default']
                gen_params_default['deps'] = deps

                # Train generative model with grid search if applicable
                gen_model, opt_b = train_model(
                    GenerativeModel,
                    L_train,
                    X_dev=L_dev,
                    Y_dev=L_gold_dev,
                    search_size=self.config['gen_model_search_space'],
                    search_params=self.config['gen_params_range'],
                    rand_seed=self.config['seed'],
                    n_threads=self.config['parallelism'],
                    verbose=self.config['verbose'],
                    params_default=gen_params_default,
                    model_init_params=self.config['gen_init_params'],
                    model_name='generative_{}'.format(self.config['domain']),
                    save_dir='checkpoints',
                    beta=self.config['gen_f_beta']
                )
                train_marginals = gen_model.marginals(L_train)

                print("\nGen. model (DP) score on dev set (b={}):".format(opt_b))
                _ = gen_model.error_analysis(self.session, L_dev, L_gold_dev, b=opt_b, display=True)

                if self.config['verbose']:
                    if self.config['display_marginals'] and not self.config['no_plots']:
                        # Display marginals
                        plt.hist(train_marginals, bins=20)
                        plt.show()
                    if self.config['display_learned_accuracies']:
                        raise NotImplementedError
                        # NOTE: Unfortunately, learned accuracies are not available after grid search
                        # lf_stats = L_dev.lf_stats(self.session, L_gold_dev, 
                        #     gen_model.learned_lf_stats()['Accuracy'])
                        # print(lf_stats)
                        # if self.config['display_correlation']:
                        #     self.display_accuracy_correlation(lf_stats)
            else:
                raise Exception("Invalid value for 'supervision': {}".format(self.config['supervision']))

        self.gen_model = gen_model
        self.L_train = L_train
        self.train_marginals = train_marginals 


    def classify(self, config=None):
        if config:
            self.config = config

        if self.config['seed']:
            np.random.seed(self.config['seed'])

        X_train = self.get_candidates(TRAIN)
        Y_train = self.train_marginals
        X_dev = self.get_candidates(DEV)
        Y_dev = load_gold_labels(self.session, annotator_name='gold', split=DEV)
        X_test = self.get_candidates(TEST)
        Y_test = load_gold_labels(self.session, annotator_name='gold', split=TEST)

        if self.config['disc_model_class'] == 'lstm':
            disc_model_class = reRNN
        else:
            raise NotImplementedError

        disc_model, opt_b = train_model(
            disc_model_class,
            X_train,
            Y_train=Y_train,
            X_dev=X_dev,
            Y_dev=Y_dev,
            cardinality=2,
            search_size=self.config['disc_model_search_space'],
            search_params=self.config['disc_params_range'],
            rand_seed=self.config['seed'],
            n_threads=self.config['parallelism'],
            verbose=self.config['verbose'],
            params_default=self.config['disc_params_default'],
            model_init_params=self.config['disc_init_params'],
            model_name='discriminative_{}'.format(self.config['domain']),
            save_dir='checkpoints',
            eval_batch_size=self.config['disc_eval_batch_size']
        )
        self.disc_model = disc_model

        scores = {}
        with PrintTimer("[7.2] Evaluate generative model (opt_b={})".format(opt_b)):
            if self.gen_model:
                # Score generative model on test set
                L_test = load_label_matrix(self.session, split=TEST)
                np.random.seed(self.config['seed'])
                scores['Gen'] = score_marginals(self.gen_model.marginals(L_test), Y_test, b=opt_b)
            else:
                print("gen_model is undefined. Skipping.")

        with PrintTimer("[7.3] Evaluate discriminative model (opt_b={})".format(opt_b)):
            # Score discriminative model trained on generative model predictions
            np.random.seed(self.config['seed'])
            scores['Disc'] = score_marginals(self.disc_model.marginals(X_test, 
                    batch_size=self.config['disc_eval_batch_size']), Y_test, b=opt_b)

        final_report(self.config, scores)


    def traditional_supervision(self, L_gold_train):
        # Confirm you have the requested number of gold labels
        train_size = self.config['max_train']
        if L_gold_train.nnz < train_size:
            print("Requested {} traditional labels. Using {} instead...".format(
                train_size, L_gold_train.nnz))
        # Randomly select the requested number of gold label
        selected = np.random.permutation(L_gold_train.nonzero()[0])[:max(train_size, L_gold_train.nnz)]
        L_train = L_gold_train[selected,:]
        train_marginals = np.array(L_train.todense()).reshape((L_train.shape[0],))
        train_marginals[train_marginals == -1] = 0
        return L_train, train_marginals


    def get_candidates(self, split):
        return self.session.query(self.candidate_class).filter(
                self.candidate_class.split == split).order_by(self.candidate_class.id).all()


    def display_accuracy_correlation(self, lf_stats):
        """Displays ..."""
        empirical = lf_stats['Empirical Acc.'].get_values()
        learned = lf_stats['Learned Acc.'].get_values()
        conflict = lf_stats['Conflicts'].get_values()
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
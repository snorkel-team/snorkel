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
from snorkel.annotations import (FeatureAnnotator, LabelAnnotator, save_marginals, 
    load_marginals, load_feature_matrix, load_label_matrix, load_gold_labels)
from snorkel.learning import GenerativeModel, MajorityVoter
from snorkel.learning.structure import DependencySelector
from snorkel.learning import reRNN, SparseLogisticRegression
from snorkel.utils import PrintTimer, ProgressBar

# Pipelines
from snorkel.contrib.babble.utils import train_model, score_marginals
from .utils import STAGES, final_report

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
        self.gen_model = None
        self.disc_model = None

        self.L_train = None
        self.L_dev = None
        self.L_test = None

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
        for stage in ['parse', 'extract', 'load_gold', 'featurize', 'collect', 'label', 
                      'supervise', 'classify']:
            stage_id = getattr(STAGES, stage.upper())
            if is_valid_stage(stage_id):
                with PrintTimer('[{}] {}...'.format(stage_id, stage.capitalize())):
                    result = getattr(self, stage)()

        return result

    def db_status(self):
        num_docs = self.session.query(Document).count()
        print("Documents: {}".format(num_docs))

        num_candidates = [0] * 3
        for split in [0,1,2]:
            num_candidates[split] = self.session.query(self.candidate_class).filter(
                self.candidate_class.split == split).count()
        print("Candidates: {}".format(num_candidates))

        # TODO:
        # How to probe the number of features and gold labels without loading?


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


    def featurize(self, config=None):
        if config:
            self.config = config
        
        if self.config['disc_model_class'] == 'lstm':
            print("Using disc_model_class='lstm'...skipping 'featurize' stage.")
            return

        featurizer = FeatureAnnotator()
        for split in self.config['splits']:
            if split == TRAIN:
                F = featurizer.apply(split=split, parallelism=self.config['parallelism'])
            else:
                F = featurizer.apply_existing(split=split, parallelism=self.config['parallelism'])
            num_candidates, num_features = F.shape
            if self.config['verbose']:
                print("Featurized split {}: ({},{}) sparse (nnz = {})".format(split, num_candidates, num_features, F.nnz))


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
            print("In 'traditional' supervision mode...skipping 'supervise' stage.")
            return                

        if self.L_train is None:
            L_train = load_label_matrix(self.session, split=TRAIN)
        else:
            L_train = self.L_train
        assert L_train.nnz > 0

        L_gold_train = load_gold_labels(self.session, annotator_name='gold', split=TRAIN)
        if self.config['verbose']:
            print("Using L_train: {0}".format(L_train.__repr__()))
            print("Using L_gold_train: {0}".format(L_gold_train.__repr__()))
            total = L_gold_train.shape[0]
            positive = float(sum(L_gold_train.todense() == 1))
            print("Positive Fraction: {:.1f}%\n".format(positive/total * 100))

        # Load DEV and TEST labels and gold labels
        if DEV in self.config['splits']:
            if self.L_dev is None:
                L_dev = load_label_matrix(self.session, split=DEV)
            else:
                L_dev = self.L_dev
            assert L_dev.nnz > 0

            L_gold_dev = load_gold_labels(self.session, annotator_name='gold', split=DEV)
            if self.config['verbose']:
                print("Using L_dev: {0}".format(L_dev.__repr__()))
                print("Using L_gold_dev: {0}".format(L_gold_dev.__repr__()))
                total = L_gold_dev.shape[0]
                positive = float(sum(L_gold_dev.todense() == 1))
                print("Positive Fraction: {:.1f}%\n".format(positive/total * 100))
            assert L_gold_dev.nonzero()[0].shape[0] > 0

        if TEST in self.config['splits']:
            if self.L_test is None:
                L_test = load_label_matrix(self.session, split=TEST)
            else:
                L_test = self.L_test            
            assert L_test.nnz > 0            

            L_gold_test = load_gold_labels(self.session, annotator_name='gold', split=TEST)
            if self.config['verbose']:
                print("Using L_test: {0}".format(L_test.__repr__()))
                print("Using L_gold_test: {0}".format(L_gold_test.__repr__()))
                total = L_gold_test.shape[0]
                positive = float(sum(L_gold_test.todense() == 1))
                print("Positive Fraction: {:.1f}%\n".format(positive/total * 100))
            assert L_gold_test.nonzero()[0].shape[0] > 0

        if self.config['supervision'] == 'majority':
            gen_model = MajorityVoter()
            train_marginals = gen_model.marginals(L_train)

        elif self.config['supervision'] == 'generative':

            # Learn dependencies
            if self.config['learn_deps']:
                if self.config['deps_thresh']:
                    ds = DependencySelector()
                    np.random.seed(self.config['seed'])
                    deps = ds.select(L_train, threshold=config['deps_thresh'])
                    if args.verbose > 0:
                        print("Selected {0} dependencies.".format(len(deps)))

                else:
                    all_deps = []
                    all_t = [0.02 + 0.02 * dep_t for dep_t in range(0, 25, 1)]

                    # Iterates over selection thresholds
                    with PrintTimer("Calculating optimal dependency threshold..."):
                        pb = ProgressBar(len(all_t))
                        for i, dep_t in enumerate(all_t):
                            pb.bar(i)
                            ds = DependencySelector()
                            deps = ds.select(L_train, propensity=True, threshold=dep_t)
                            all_deps.append(deps)

                            if len(deps) == 0:
                                break
                        pb.close()

                    # Selects point of approximate maximum curvature
                    max_curvature = float('-Inf')
                    i_max = None
                    for i in range(1, len(all_deps) - 1):
                        curvature = len(all_deps[i+1]) + len(all_deps[i-1]) \
                                    - 2 * len(all_deps[i])
                        if curvature > max_curvature:
                            max_curvature = curvature
                            i_max = i

                    deps = all_deps[i_max]
                    print("Selected threshold {0} for {1} dependencies."
                            .format(all_t[i_max], len(deps)))

            else:
                deps = ()

            # Pass in the dependencies via default params
            gen_params_default = self.config['gen_params_default']
            gen_params_default['deps'] = list(deps)

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
                save_dir=os.path.join(self.config['reports_dir'], 'checkpoints'),
                beta=self.config['gen_f_beta'],
                tune_b=self.config['tune_b'],
            )
            train_marginals = gen_model.marginals(L_train)

            print("\nGen. model (DP) score on dev set (b={}):".format(opt_b))
            tp, fp, tn, fn = gen_model.error_analysis(self.session, L_dev, L_gold_dev, b=opt_b, display=True)
            
            # Record generative model performance
            precision = float(len(tp))/float(len(tp) + len(fp)) if len(tp) + len(fp) else 0
            recall = float(len(tp))/float(len(tp) + len(fn)) if len(tn) + len(fn) else 0
            f1 = float(2 * precision * recall)/(precision + recall) if (precision or recall) else 0
            self.scores = {}
            self.scores['Gen'] = [precision, recall, f1]

            if self.config['verbose']:
                if self.config['display_marginals'] and not self.config['no_plots']:
                    # Display marginals
                    plt.hist(train_marginals, bins=20)
                    plt.show()
                # if self.config['display_learned_accuracies']:
                    # raise NotImplementedError
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
        save_marginals(self.session, L_train, train_marginals)

        if (self.config['supervision'] == 'generative' and 
            self.config['end_at'] == STAGES.CLASSIFY):
            final_report(self.config, self.scores)


    def classify(self, config=None):
        if config:
            self.config = config

        if self.config['seed']:
            np.random.seed(self.config['seed'])

        Y_dev = load_gold_labels(self.session, annotator_name='gold', split=DEV)
        Y_test = load_gold_labels(self.session, annotator_name='gold', split=TEST)

        if self.config['disc_model_class'] == 'lstm':
            disc_model_class = reRNN

            if self.config['supervision'] == 'traditional':

                print("In 'traditional' supervision mode...grabbing candidate and gold label subsets.")  
                if self.config['traditional_split'] != TRAIN:
                    print("NOTE: using split {} for traditional supervision. "
                        "Be aware of unfair evaluation.".format(self.config['traditional_split']))
                
                candidates = self.get_candidates(split=self.config['traditional_split'])
                L_gold = load_gold_labels(self.session, annotator_name='gold', 
                                          split=self.config['traditional_split'])

                Y_train = np.array(L_gold.todense()).reshape((L_gold.shape[0],))
                Y_train[Y_train == -1] = 0

                X_train, Y_train = self.traditional_supervision(candidates, Y_train)
            else:
                X_train = self.get_candidates(TRAIN)
                Y_train = (self.train_marginals if getattr(self, 'train_marginals', None) is not None 
                    else load_marginals(self.session, split=0))  

            X_dev = self.get_candidates(DEV)
            X_test = self.get_candidates(TEST)

        elif self.config['disc_model_class'] == 'logreg':
            disc_model_class = SparseLogisticRegression

            if self.config['supervision'] == 'traditional':
                
                print("In 'traditional' supervision mode...grabbing candidate and gold label subsets.")  
                if self.config['traditional_split'] != TRAIN:
                    print("NOTE: using split {} for traditional supervision. "
                        "Be aware of unfair evaluation.".format(self.config['traditional_split']))
                
                X_train = load_feature_matrix(self.session, 
                                              split=self.config['traditional_split'])
                L_gold = load_gold_labels(self.session, annotator_name='gold', 
                                          split=self.config['traditional_split'])


                Y_train = np.array(L_gold.todense()).reshape((L_gold.shape[0],))
                Y_train[Y_train == -1] = 0

                X_train, Y_train = self.traditional_supervision(X_train, Y_train)
            else:
                X_train = load_feature_matrix(self.session, split=TRAIN)
                Y_train = (self.train_marginals if getattr(self, 'train_marginals', None) is not None 
                    else load_marginals(self.session, split=0))

            X_dev = load_feature_matrix(self.session, split=DEV)
            X_test = load_feature_matrix(self.session, split=TEST)
        else:
            raise NotImplementedError

        print(Y_train.shape)
        if self.config['display_marginals'] and not self.config['no_plots']:
            plt.hist(Y_train, bins=20)
            plt.show()

        with PrintTimer("[7.1] Begin training discriminative model"):
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
                save_dir=os.path.join(self.config['reports_dir'], 'checkpoints'),
                eval_batch_size=self.config['disc_eval_batch_size'],
                tune_b=self.config['tune_b'],
            )
        self.disc_model = disc_model

        self.scores = {}
        with PrintTimer("[7.2] Evaluate generative model (opt_b={})".format(opt_b)):
            if self.gen_model is not None:
                # Score generative model on test set
                if self.L_test is None:
                    L_test = load_label_matrix(self.session, split=TEST)
                else:
                    L_test = self.L_test            
                assert L_test.nnz > 0         

                np.random.seed(self.config['seed'])
                self.scores['Gen'] = score_marginals(
                    self.gen_model.marginals(L_test), Y_test, b=opt_b)
            else:
                print("gen_model is undefined. Skipping.")

        with PrintTimer("[7.3] Evaluate discriminative model (opt_b={})".format(opt_b)):
            # Score discriminative model trained on generative model predictions
            np.random.seed(self.config['seed'])
            self.scores['Disc'] = score_marginals(self.disc_model.marginals(X_test, 
                    batch_size=self.config['disc_eval_batch_size']), Y_test, b=opt_b)

        final_report(self.config, self.scores)


    def traditional_supervision(self, X, Y):
        """Extract a specified number of candidates with labels.
        Args:
            X: csr_AnnotationMatrix of features or list of candidates (for entire split)
            Y: np.array of marginals [0, 1]
        Returns:
            X_out: extracted csr_AnnotationMatrix or candidates from X
            Y_out: extracted np.array of marginals from Y
        """
        print("NOTE: traditional supervision helper assumes all candidates have labels.")
        
        # Confirm you have the requested number of gold labels
        train_size = self.config['max_train']
        total_size = len(Y)
        if isinstance(X, list):
            assert(len(X) == total_size)
        else:
            assert(X.shape[0] == total_size)

        if train_size is None:
            print("No value found for max_train. Using all available gold labels.")
            train_size = total_size
        elif total_size < train_size:
            print("Requested {} traditional labels. Only {} could be found.".format(
                train_size, total_size))
            train_size = total_size
        print("Using {} traditional gold labels".format(train_size))
        
        if train_size == total_size:
            return X, Y

        # Randomly select the requested number of candidates + gold labels
        selected = sorted(np.random.permutation(total_size)[:train_size])
        Y_out = Y[selected]
        if isinstance(X, list):
            X_out = [c for i, c in enumerate(X) if i in set(selected)]
        else:
            X_out = X[selected]
        return X_out, Y_out


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


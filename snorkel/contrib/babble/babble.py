from collections import defaultdict, namedtuple
import random

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
import scipy.sparse as sparse

from snorkel.annotations import LabelAnnotator, load_gold_labels, csr_AnnotationMatrix
from snorkel.learning.utils import MentionScorer
from snorkel.lf_helpers import test_LF
from snorkel.utils import (
    matrix_conflicts,
    matrix_coverage,
    matrix_overlaps,
    matrix_tp,
    matrix_fp,
    matrix_fn,
    matrix_tn
)

from snorkel.contrib.babble.filter_bank import FilterBank
from snorkel.contrib.babble.grammar import Parse
from snorkel.contrib.babble.semparser import Explanation, SemanticParser


ConfusionMatrix = namedtuple('ConfusionMatrix', ['correct', 'incorrect', 'abstained'])
Metrics = namedtuple('Metrics', ['accuracy', 'coverage', 'class_coverage'])

class Statistic(object):
    def __init__(self, numer, denom):
        self.numer = int(numer)
        self.denom = int(denom)
        self.percent = float(numer)/denom * 100 if denom else 0
    
    def __repr__(self):
        return "{}: {:.2f}% ({}/{})".format(
            type(self).__name__, self.percent, self.numer, self.denom)

class Accuracy(Statistic):
    """The accuracy of a single labeling function."""
    pass

class Coverage(Statistic):
    """The coverage (# labeled/# total) of a single labeling function."""
    pass

class ClassCoverage(Statistic):
    """The coverage (# labeled/# total _in class_) of a single labeling function."""
    pass

class GlobalCoverage(Statistic):
    """The coverage (# labeled with 1+ label/# total) of all labeling functions."""
    pass


class CandidateGenerator(object):
    """
    A generator for returning a list of candidates in a certain order.
    """
    def __init__(self, babble_stream, seed=None, 
                 balanced=False, active=False, shuffled=False):
        """
        If active = True, return only candidates that have no labels so far
        If balanced = True, alternate between candidates with True/False gold labels
        If random = True, return the candidates (passing the above conditions,
            if applicable) in random order.
        """
        candidates = babble_stream.dev_candidates
        labels = babble_stream.dev_labels
        
        if active:
            raise NotImplementedError
        else:
            if balanced:
                self.candidate_generator = self.balanced_generator(
                    candidates, labels, seed, shuffled=shuffled)
            else:
                self.candidate_generator = self.linear_generator(
                    candidates, seed, shuffled=shuffled)

    def __iter__(self):
        return self

    def next(self):
        return self.candidate_generator.next()

    @staticmethod
    def linear_generator(candidates, seed, shuffled=False):
        if shuffled:
            if seed is not None:
                random.seed(seed)
            random.shuffle(candidates)
        for c in candidates:
            yield c

    @staticmethod
    def balanced_generator(candidates, labels, seed, shuffled=False):
        candidates_labels = zip(candidates, labels)
        if shuffled:
            if seed is not None:
                random.seed(seed)
            random.shuffle(candidates_labels)
        positives = [c for (c, l) in candidates_labels if l == 1]
        negatives = [c for (c, l) in candidates_labels if l == -1]
        candidate_queue = []
        for i in range(max(len(positives), len(negatives))):
            if i < len(positives):
                candidate_queue.append(positives[i])
            if i < len(negatives):
                candidate_queue.append(negatives[i])
        for c in candidate_queue:
            yield c


class BabbleStream(object):
    """
    An object for iteratively viewing candidates and parsing corresponding explanations.
    """
    def __init__(self, session, mode='text', candidate_class=None, seed=None, 
                 user_lists={}, verbose=True, **kwargs):
        self.session = session
        self.mode = mode
        self.candidate_class = candidate_class
        self.seed = seed
        self.verbose = verbose

        self.dev_candidates = session.query(self.candidate_class).filter(self.candidate_class.split == 1).all()
        self.dev_labels = np.ravel((load_gold_labels(session, annotator_name='gold', split=1)).todense())
    
        self.candidate_generator_kwargs = kwargs
        self.user_lists = user_lists
        self.semparser = None
        self.filter_bank = FilterBank(session, candidate_class)
        
        self.parses = []
        self.label_matrix = None

        # Temporary storage
        # self.temp_explanations = None
        self.temp_parses = None
        self.temp_label_matrix = None

        # Evaluation tools
        self.num_dev_total  = len(self.dev_candidates)
        self.num_dev_pos    = sum(self.dev_labels == 1)
        self.num_dev_neg    = sum(self.dev_labels == -1)
        assert(self.num_dev_total == self.num_dev_pos + self.num_dev_neg)
        self.scorer         = MentionScorer(self.dev_candidates, self.dev_labels)


    def __iter__(self):
        return self

    def next(self):
        if not hasattr(self, 'candidate_generator'):
            self.candidate_generator = CandidateGenerator(self,
                seed=self.seed, **(self.candidate_generator_kwargs))
        c = self.candidate_generator.next()
        self.temp_candidate = c
        return c

    def _build_semparser(self):
        self.semparser = SemanticParser(
            mode=self.mode, candidate_class=self.candidate_class, 
            user_lists=self.user_lists) #top_k=-4, beam_width=30)

    def add_user_lists(self, new_user_lists):
        """
        Adds additional user_lists and rebuilds SemanticParser.
        
        :param new_user_lists: A dict {k: v, ...}
            k = (string) list name
            v = (list) words belonging to the user_list
        """
        self.user_lists.update(new_user_lists)
        self._build_semparser()

    def preload(self, explanations=None, user_lists=None):
        """
        Load and commit the provided user_lists and/or explanations.
        """
        if user_lists:
            self.add_user_lists(user_lists)
        if explanations:
            parses, _, _, _ = self.apply(explanations)
            if parses:
                self.commit()

    def apply(self, explanations, split=1, parallelism=1):
        """
        :param explanations: an Explanation or list of Explanations.
        :param parallelism: number of threads to use; CURRENTLY UNUSED
        :param split: the split to use for the filter bank; CURRENTLY UNUSED
        """
        self.commit([])
        print("All previously uncommitted parses have been flushed.")

        if parallelism != 1:
            raise NotImplementedError("BabbleStream does not yet support parallelism > 1")

        if split != 1:
            raise NotImplementedError("BabbleStream does not yet support splits != 1")

        parses = self._parse(explanations)
        parses, filtered_parses, label_matrix = self._filter(parses, explanations)
        conf_matrix_list, stats_list = self.analyze(parses)
        
        # Hold results in temporary space until commit
        # self.temp_explanations = explanations if isinstance(explanations, list) else [explanations]
        self.temp_parses = parses if isinstance(parses, list) else [parses]
        self.temp_label_matrix = label_matrix
        
        return parses, filtered_parses, conf_matrix_list, stats_list

    def _parse(self, explanations):
        """
        :param explanations: an Explanation or list of Explanations.
        :return: a list of Parses.
        """
        if not self.semparser:
            self._build_semparser()

        parses = self.semparser.parse(explanations, 
            return_parses=True, verbose=self.verbose)

        return parses
    
    def _filter(self, parses, explanations):
        """
        :param parses: a Parse or list of Parses.
        :param explanations: the Explanation or list of Explanations from which 
            the parse(s) were produced.
        :return: the outputs from filter_bank.apply()
        """
        return self.filter_bank.apply(parses, explanations)
        

    def analyze(self, parses):
        if not parses:
            return [], []

        # TODO: improve the efficiency of this by applying labeler object in
        # parallel mode and extracting stats from label_matrix.
        conf_matrix_list = []
        stats_list = []
        for parse in parses:
            lf = parse.function
            dev_marginals  = np.array([0.5 * (lf(c) + 1) for c in self.dev_candidates])
            tp, fp, tn, fn = self.scorer.score(dev_marginals, 
                set_unlabeled_as_neg=False, set_at_thresh_as_neg=False, display=False)

            TP, FP, TN, FN = map(lambda x: len(x), [tp, fp, tn, fn])
            if TP or FP:
                conf_matrix = (ConfusionMatrix(tp, fp, set(self.dev_candidates) - tp - fp))
                accuracy = Accuracy(TP, TP + FP)
                coverage = Coverage(TP + FP, self.num_dev_total)
                class_coverage = ClassCoverage(TP + FP, self.num_dev_pos)
            elif TN or FN:
                conf_matrix = (ConfusionMatrix(tn, fn, set(self.dev_candidates) - tn - fn))
                accuracy = Accuracy(TN, TN + FN)
                coverage = Coverage(TN + FN, self.num_dev_total)
                class_coverage = ClassCoverage(TN + FN, self.num_dev_neg)
            else:
                conf_matrix = ConfusionMatrix(set(), set(), set(self.dev_candidates))
                accuracy = Accuracy(0, self.num_dev_total)
                coverage = Coverage(0, self.num_dev_total)
                class_coverage = ClassCoverage(0, self.num_dev_total)

            conf_matrix_list.append(conf_matrix)
            stats_list.append(Metrics(accuracy, coverage, class_coverage))

        return conf_matrix_list, stats_list

    def filtered_analysis(self, filtered_parses):
        if not any(filtered_parses.values()):
            print("No filtered parses to analyze.")
            return
        for filter_name, parses in filtered_parses.items():
            if parses:
                print("Filter {} removed {} parse(s):".format(filter_name, len(parses)))
            for i, filtered_parse in enumerate(parses):
                print("\n#{} Filtered parse:".format(i))
                print("Explanation (source):\n{}".format(
                    filtered_parse.parse.explanation))
                print("\nParse (pseudocode):\n{}".format(
                    self.semparser.grammar.translate(filtered_parse.parse.semantics)))

                if filter_name == 'DuplicateSemanticsFilter':
                    print("\nReason:\nCollision with parse from this explanation:\n{}".format(
                        filtered_parse.reason.explanation))
                    
                elif filter_name == 'ConsistencyFilter':
                    candidate = filtered_parse.reason
                    print('\nReason:\nInconsistent with candidate ({}, {}) from:\n"{}"'.format(
                        candidate[0].get_span(), candidate[1].get_span(), 
                        filtered_parse.reason.get_parent().text))
                    
                elif filter_name == 'UniformSignatureFilter':
                    print("\nReason:\n{}".format(filtered_parse.reason))
                    
                elif filter_name == 'DuplicateSignatureFilter':
                    print("\nReason:\nCollision with parse from this explanation:\n{}".format(
                        filtered_parse.reason.explanation))


    def commit(self, idxs='all'):
        """
        :param idxs: The indices of the parses (from the most recently returned
            list of parses) to permanently keep. 
            If idxs = 'all', keep all of the parses.
            If idxs is an integer, keep just that one parse.
            If idxs is a list of integers, keep all parses from that list.
            If idxs = None or [], keep none of the parses.
        """
        if idxs == 'all':
            idxs = range(len(self.temp_parses))
        elif isinstance(idxs, int):
            idxs = [idxs]
        elif not idxs:
            idxs = []
            print("Flushing all parses from previous explanation set.")
        
        if (isinstance(idxs, list) and len(idxs) > 0 and 
            all(isinstance(x, int) for x in idxs)):
            if max(idxs) >= len(self.temp_parses):
                raise Exception("Invalid idx: {}.".format(max(idxs)))

            parses_to_add = [p for i, p in enumerate(self.temp_parses) if i in idxs]
            parse_names_to_add = [p.function.__name__ for p in parses_to_add]
            explanations_to_add = set([parse.explanation for parse in parses_to_add])
      
            self.parses.extend(parses_to_add)
            if self.label_matrix is None:
                self.label_matrix = self.temp_label_matrix
            else:
                self.label_matrix = sparse.hstack((self.label_matrix, self.temp_label_matrix))

            if self.verbose:
                print("Added {} parse(s) from {} explanations to set. (Total # parses = {})".format(
                    len(parses_to_add), len(explanations_to_add), len(self.parses)))

        # Permanently store the semantics and signatures in duplicate filters
        self.filter_bank.commit(idxs)

        self.temp_parses = None
        self.temp_label_matrix = None

    def get_global_coverage(self):
        """Calculate stats for the dataset as a whole.

        Note: this only consideres committed LFs
        TODO: use sparse_abs from snorkel.utils here and elsewhere.
        """
        num_labeled = sum(np.asarray(abs(np.sum(self.label_matrix, 1))).ravel() != 0)

        return GlobalCoverage(num_labeled, self.num_dev_total)

    # def _sparse_to_csr_annotation_matrix(self, label_matrix)
    #     """Convert a sparse label_matrix into a csr_AnnotationMatrix.
        
    #     Note: assumes it will only be applied to dev_candidates.
    #     """
    #     candidate_index = {c.id: i for i, c in enumerate(self.dev_candidates)}
    #     row_index = {v: k for k, v in candidate_index.iteritems()}
    #     key_index = 

    def get_parses(self, idx=None, translate=True):
        if idx is None:
            parses = self.parses
        elif isinstance(idx, int):
            parses = [self.parses[idx]]
        elif isinstance(idx, list):
            parses = [parse for i, parse in enumerate(self.parses) if i in idx]
        
        if translate:
            return [self.semparser.grammar.translate(parse.semantics) for parse in parses]
        else:
            return parses

    def get_lfs(self, idx=None):
        return [parse.function for parse in self.get_parses(idx=idx, translate=False)]

    def get_explanations(self, idx=None):
        explanations = []
        explanations_set = set()
        for parse in self.get_parses(idx=idx, translate=False):
            explanation = parse.explanation
            if explanation not in explanations_set:
                explanations.append(explanation)
                explanations_set.add(explanation)
        return explanations

    def get_lf_stats(self):
        """Returns a pandas DataFrame with the LFs and various per-LF statistics.
        
        NOTE: assumes you're asking about the dev set.
        """
        label_matrix = sparse.csr_matrix(self.label_matrix)
        labels = self.dev_labels
        lf_names = [parse.function.__name__ for parse in self.parses]

        # Default LF stats
        col_names = ['j', 'Coverage', 'Overlaps', 'Conflicts']
        d = {
            'j'         : range(label_matrix.shape[1]),
            'Coverage'  : Series(data=matrix_coverage(label_matrix), index=lf_names),
            'Overlaps'  : Series(data=matrix_overlaps(label_matrix), index=lf_names),
            'Conflicts' : Series(data=matrix_conflicts(label_matrix), index=lf_names)
        }
        col_names.extend(['TP', 'FP', 'FN', 'TN', 'Empirical Acc.'])
        tp = matrix_tp(label_matrix, labels)
        fp = matrix_fp(label_matrix, labels)
        fn = matrix_fn(label_matrix, labels)
        tn = matrix_tn(label_matrix, labels)
        ac = (tp+tn).astype(float) / (tp+tn+fp+fn)
        d['Empirical Acc.'] = Series(data=ac, index=lf_names)
        d['TP']             = Series(data=tp, index=lf_names)
        d['FP']             = Series(data=fp, index=lf_names)
        d['FN']             = Series(data=fn, index=lf_names)
        d['TN']             = Series(data=tn, index=lf_names)

        return DataFrame(data=d, index=lf_names)[col_names]
        

    def get_label_matrix(self):
        if self.temp_parses is not None:
            print("You must commit before retrieving the label matrix.")
            return None
        # TODO: For now, return a csr_matrix. Later, confirm we don't need to convert.
        # label_matrix = csr_LabelMatrix(
        #     self.label_matrix, 
        #     candidate_index=
        #     row_index=
        #     annotation_key_cls=
        #     key_index=
        #     col_index=)
        return self.label_matrix


class Babbler(BabbleStream):
    def apply(self, *args, **kwargs):
        BabbleStream.apply(self, *args, **kwargs)
        self.commit()
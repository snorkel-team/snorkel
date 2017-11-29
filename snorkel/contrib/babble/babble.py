from collections import defaultdict, namedtuple
import itertools
import random

import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame, Series
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sparse

from snorkel.annotations import LabelAnnotator, load_gold_labels, csr_AnnotationMatrix
from snorkel.learning import MajorityVoter
from snorkel.learning.utils import MentionScorer
from snorkel.lf_helpers import test_LF
from snorkel.utils import (
    PrintTimer,
    ProgressBar,
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
from snorkel.contrib.babble.utils import score_marginals


ConfusionMatrix = namedtuple('ConfusionMatrix', ['correct', 'incorrect', 'abstained'])
Metrics = namedtuple('Metrics', ['accuracy', 'coverage', 'class_coverage'])
# Use 'parse' as field instead of 'explanation' to match with FilteredParse object.
FilteredExplanation = namedtuple('FilteredExplanation', ['parse', 'reason'])

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
                 balanced=False, active=False, shuffled=False,
                 priority_candidate_ids=[]):
        """
        If active = True, return only candidates that have no labels so far
        If balanced = True, alternate between candidates with True/False gold labels
        If random = True, return the candidates (passing the above conditions,
            if applicable) in random order.
        """
        candidates = babble_stream.dev_candidates
        labels = babble_stream.dev_labels
        
        candidates, labels, priority_generator = self.make_priority_generator(
            candidates, labels, priority_candidate_ids)
        self.priority_generator = priority_generator

        if active:
            raise NotImplementedError
        else:
            if balanced:
                self.candidate_generator = itertools.chain(
                    priority_generator, self.balanced_generator(
                        candidates, labels, seed, shuffled=shuffled))
            else:
                self.candidate_generator = itertools.chain(
                    priority_generator, self.linear_generator(
                        candidates, labels, seed, shuffled=shuffled))

    def __iter__(self):
        return self

    def next(self):
        return self.candidate_generator.next()

    def make_priority_generator(self, candidates, labels, priority_candidate_ids):
        # Pull out priority candidates if applicable
        # Go for the slightly more wasteful but easy-to-understand solution

        if priority_candidate_ids:
            def simple_generator(candidates):
                for c in candidates:
                    yield c

            priority_set = set(priority_candidate_ids)
            priority = []
            other = []

            # Pull out all priority candidates
            for c, l in zip(candidates, labels):
                if c.get_stable_id() in priority_set:
                    priority.append(c)
                else:
                    # Hold on to the labels for a possible balanced_generator downstream
                    other.append((c, l))
            # Put them in desired order
            priority_idxs = {c: i for i, c in enumerate(priority_candidate_ids)}
            priority.sort(key=lambda x: priority_idxs[x.get_stable_id()])
            priority_generator = simple_generator(priority)
            # Restore remaining candidates and labels to normal lists
            candidates, labels = zip(*other)
        else:
            priority_generator = iter(())

        return candidates, labels, priority_generator

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
                 user_lists={}, apply_filters=True, verbose=True, 
                 soft_start=False, **kwargs):
        self.session = session
        self.mode = mode
        self.candidate_class = candidate_class
        self.seed = seed
        self.verbose = verbose
        self.apply_filters = apply_filters

        self.dev_candidates = session.query(self.candidate_class).filter(self.candidate_class.split == 1).all()
        self.dev_labels = np.ravel((load_gold_labels(session, annotator_name='gold', split=1)).todense())
    
        # self.candidate_generator_kwargs = kwargs
        ### TEMP HARDCODE ###
        if soft_start:
            priority_ids = [
                '7fc3e510-c4e6-44c2-a24b-f9a39bfcfb07::span:4942:4950~~7fc3e510-c4e6-44c2-a24b-f9a39bfcfb07::span:4973:4978',
                '40cb15fa-0186-4868-a5b7-eb2fc6a317cf::span:115:123~~40cb15fa-0186-4868-a5b7-eb2fc6a317cf::span:138:153',
                '7fc3e510-c4e6-44c2-a24b-f9a39bfcfb07::span:1926:1945~~7fc3e510-c4e6-44c2-a24b-f9a39bfcfb07::span:1956:1968',
            ]
        else:
            priority_ids = []

        self.candidate_generator = CandidateGenerator(self, seed=self.seed, 
            priority_candidate_ids=priority_ids, **kwargs)
        ### TEMP HARDCODE ###
            
        self.user_lists = user_lists
        self.semparser = None
        self.filter_bank = FilterBank(session, candidate_class)
        
        self.parses = []
        self.label_matrix = None
        # rows, cols, data, shape:(_, _)
        self.label_triples = [[[],[],[],0,0], None, [[],[],[],0,0]]

        # Temporary storage
        self.temp_parses = None
        self.temp_label_matrix = None
        self.last_parses = []

        # Evaluation tools
        self.num_dev_total  = len(self.dev_candidates)
        self.num_dev_pos    = sum(self.dev_labels == 1)
        self.num_dev_neg    = sum(self.dev_labels == -1)
        if self.num_dev_pos + self.num_dev_neg != self.num_dev_total:
            print("WARNING: Number of candidates ({}) does not equal the number "
                "of pos ({}) + neg ({}) = {} labels.".format(
                    self.num_dev_total, self.num_dev_pos, self.num_dev_neg,
                    self.num_dev_pos + self.num_dev_neg))
        self.scorer         = MentionScorer(self.dev_candidates, self.dev_labels)

    def __iter__(self):
        return self


    def next(self):
        c = self.candidate_generator.next()
        self.temp_candidate = c
        return c

    def _build_semparser(self):
        self.semparser = SemanticParser(
            mode=self.mode, candidate_class=self.candidate_class, 
            user_lists=self.user_lists, beam_width=10) #top_k=-4, beam_width=30)

    def add_user_lists(self, new_user_lists):
        """
        Adds additional user_lists and rebuilds SemanticParser.
        
        :param new_user_lists: A dict {k: v, ...}
            k = (string) list name
            v = (list) words belonging to the user_list
        """
        self.user_lists.update(new_user_lists)
        self._build_semparser()

    def preload(self, explanations=None, user_lists=None, label_others=True):
        """
        Load and commit the provided user_lists and/or explanations.
        """
        if user_lists:
            self.add_user_lists(user_lists)
        if explanations:
            parses, _, _, _ = self.apply(explanations)
            if parses:
                self.commit()
            # Also label train and test
            # if label_others:
            #     self.label_split(0)
            #     self.label_split(2)

    def apply(self, explanations, split=1, parallelism=1):
        """
        :param explanations: an Explanation or list of Explanations.
        :param parallelism: number of threads to use; CURRENTLY UNUSED
        :param split: the split to use for the filter bank; CURRENTLY UNUSED
        """
        # Flush all uncommmitted results from previous runs
        self.commit([])

        if parallelism != 1:
            raise NotImplementedError("BabbleStream does not yet support parallelism > 1")

        if split != 1:
            raise NotImplementedError("BabbleStream does not yet support splits != 1")

        explanations = explanations if isinstance(explanations, list) else [explanations]
        parses, unparseable_explanations = self._parse(explanations)
        if self.apply_filters:
            parses, filtered_parses, label_matrix = self._filter(parses, explanations)
        else:
            print("Because apply_filters=False, no parses are being filtered.")
            filtered_parses = {}
            label_matrix = self.filter_bank.label(parses)
        conf_matrix_list, stats_list = self.analyze(parses)

        filtered_objects = filtered_parses
        filtered_objects['UnparseableExplanations'] = unparseable_explanations
        
        # Hold results in temporary space until commit
        # self.temp_explanations = explanations if isinstance(explanations, list) else [explanations]
        self.temp_parses = parses if isinstance(parses, list) else [parses]
        self.temp_label_matrix = label_matrix
        self.temp_filtered_objects = filtered_objects

        return parses, filtered_objects, conf_matrix_list, stats_list

    def _parse(self, explanations):
        """
        :param explanations: an Explanation or list of Explanations.
        :return: a list of Parses.
        """
        if not self.semparser:
            self._build_semparser()

        parses = self.semparser.parse(explanations, 
            return_parses=True, verbose=self.verbose)
        used_explanations = set([p.explanation for p in parses])
        unparseable_explanations = [FilteredExplanation(exp, 'Unparseable') 
            for exp in explanations if exp not in used_explanations]

        return parses, unparseable_explanations
    
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
                class_coverage = ClassCoverage(TP, self.num_dev_pos)
            elif TN or FN:
                conf_matrix = (ConfusionMatrix(tn, fn, set(self.dev_candidates) - tn - fn))
                accuracy = Accuracy(TN, TN + FN)
                coverage = Coverage(TN + FN, self.num_dev_total)
                class_coverage = ClassCoverage(TN, self.num_dev_neg)
            else:
                conf_matrix = ConfusionMatrix(set(), set(), set(self.dev_candidates))
                accuracy = Accuracy(0, self.num_dev_total)
                coverage = Coverage(0, self.num_dev_total)
                class_coverage = ClassCoverage(0, self.num_dev_total)

            conf_matrix_list.append(conf_matrix)
            stats_list.append(Metrics(accuracy, coverage, class_coverage))

        return conf_matrix_list, stats_list

    def filtered_analysis(self, filtered_parses=None):
        if filtered_parses is None:
            # Use the last set of filtered parses to be produced.
            filtered_parses = self.temp_filtered_objects

        if filtered_parses is None or not any(filtered_parses.values()):
            print("No filtered parses to analyze.")
            return

        filter_names = [
            'UnparseableExplanations',
            'DuplicateSemanticsFilter',
            'ConsistencyFilter',
            'UniformSignatureFilter',
            'DuplicateSignatureFilter',
        ]

        num_filtered = 0
        print("SUMMARY")
        print("{} TOTAL:".format(
            sum([len(p) for p in filtered_parses.values()])))
        print("{} Unparseable Explanation".format(
            len(filtered_parses.get('UnparseableExplanations', []))))
        print("{} Duplicate Semantics".format(
            len(filtered_parses.get('DuplicateSemanticsFilter', []))))
        print("{} Inconsistency with Example".format(
            len(filtered_parses.get('ConsistencyFilter', []))))
        print("{} Uniform Signature".format(
            len(filtered_parses.get('UniformSignatureFilter', []))))
        print("{} Duplicate Signature".format(
            len(filtered_parses.get('DuplicateSignatureFilter', []))))

        for filter_name in filter_names:
            parses = filtered_parses.get(filter_name, [])

            for filtered_parse in parses:
                num_filtered += 1

                if filtered_parse.reason == 'Unparseable':
                    parse_str = filtered_parse.parse.condition
                else:
                    parse_str = self.semparser.grammar.translate(filtered_parse.parse.semantics)

                if filter_name == 'UnparseableExplanations':
                    filter_str = "Unparseable Explanation"
                    reason_str = "This explanation couldn't be parsed."

                elif filter_name == 'DuplicateSemanticsFilter':
                    filter_str = "Duplicate Semantics"
                    reason_str = 'This parse is identical to one produced by the following explanation:\n\t"{}"'.format(
                        filtered_parse.reason.explanation.condition)
                    
                elif filter_name == 'ConsistencyFilter':
                    candidate = filtered_parse.reason
                    filter_str = "Inconsistency with Example"
                    reason_str = "This parse did not agree with the candidate ({}, {})".format(
                        candidate[0].get_span(), candidate[1].get_span())
                        # filtered_parse.reason.get_parent().text.encode('utf-8')))
                    
                elif filter_name == 'UniformSignatureFilter':
                    filter_str = "Uniform Signature"
                    reason_str = "This parse labeled {} of the {} development examples".format(
                        filtered_parse.reason, self.num_dev_total)
                    
                elif filter_name == 'DuplicateSignatureFilter':
                    filter_str = "Duplicate Signature"
                    reason_str = "This parse labeled identically to the following existing parse:\n\t{}".format(
                        self.semparser.grammar.translate(filtered_parse.reason.explanation))

                print("\n[#{}]: {}".format(num_filtered, filter_str))
                # print("\nFilter: {}".format(filter_str))
                if filtered_parse.reason == 'Unparseable':
                    print("\nExplanation: {}".format(parse_str))
                else:
                    print("\nParse: {}".format(parse_str))
                print("\nReason: {}\n".format(reason_str))


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

            self.last_parses = parses_to_add
            if self.verbose:
                print("Added {} parse(s) from {} explanations to set. (Total # parses = {})".format(
                    len(parses_to_add), len(explanations_to_add), len(self.parses)))

        # Permanently store the semantics and signatures in duplicate filters
        self.filter_bank.commit(idxs)                

        self.temp_parses = None
        self.temp_label_matrix = None

    def label_split(self, split):
        """Label a single split with the most recently committed LFs."""
        if split == 1:
            raise Exception("The dev set is labeled during Babbler.apply() by the FilterBank.")

        with PrintTimer("Applying labeling functions to split {}".format(split)):
            lfs = [parse.function for parse in self.last_parses]
            candidates = self.session.query(self.candidate_class).filter(
                self.candidate_class.split == split).all()
            num_existing_lfs = self.label_triples[split][4]

            rows = []
            cols = []
            data = []
            pb = ProgressBar(len(candidates) * len(lfs))
            count = 0
            for j, lf in enumerate(lfs):
                for i, c in enumerate(candidates):
                    pb.bar(count)
                    count += 1
                    label = lf(c)
                    if label:
                        rows.append(i)
                        cols.append(j + num_existing_lfs)
                        data.append(label)
            pb.close()
            # NOTE: There is potential for things to go wrong if the user calls
            # this function twice and the label matrix ends up wonky.
            self.label_triples[split][0].extend(rows)
            self.label_triples[split][1].extend(cols)
            self.label_triples[split][2].extend(data)
            self.label_triples[split][3] = len(candidates)
            self.label_triples[split][4] += len(lfs)
            print("Stored {} triples for split {}. Now shape is ({}, {}).".format(
                len(data), split, self.label_triples[split][3], self.label_triples[split][4]))

    def get_labeled_equivalent(self, f1):
        """Returns the number of ground truth labels required for the given f1.
        
        F1 curve is based on a curve fit to empirical results on an LSTM on
        the spouse domain.
        """
        if f1 < 0.12:
            return str(len(self.get_explanations()))
        elif f1 > 0.5:
            return str("20,000+")
        else:
            # y = ax^2 + bx + c
            a = -1.27e-9
            b = 4.48e-5
            c = 0.121 - f1
            return str(int((-b + np.sqrt(b*b - 4*a*c))/float(2*a)))

    def get_majority_quality(self, split=1):
        """Calculates the quality on the dev set using simple majority vote."""
        majority_voter = MajorityVoter()

        L_split = self.get_label_matrix(split=split)
        if not L_split.nnz:
            print("Cannot calculate majority quality for split {} because label "
                "matrix is empty.".format(split))
            return None
        
        Y_split = load_gold_labels(self.session, annotator_name='gold', split=split)
        
        pr, re, f1, cov = score_marginals(majority_voter.marginals(L_split), Y_split)
        return (f1, pr, re)

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

    def get_label_matrix(self, split=1):
        if split == 1:
            if self.temp_parses is not None:
                print("You must commit before retrieving the label matrix.")
                return None
            label_matrix = self.label_matrix
        else:
            rows, cols, data, shape_row, shape_col = self.label_triples[split]
            label_matrix = coo_matrix((data, (rows, cols)), shape=(shape_row, shape_col)).tocsr()
        
        candidates = self.session.query(self.candidate_class).filter(
            self.candidate_class.split == split).all()
        candidate_index = {c.id: i for i, c in enumerate(candidates)}
        row_index = {v: k for k, v in candidate_index.items()}
        return csr_AnnotationMatrix(label_matrix, 
                                    candidate_index=candidate_index,
                                    row_index=row_index)


class Babbler(BabbleStream):
    def apply(self, *args, **kwargs):
        BabbleStream.apply(self, *args, **kwargs)
        self.commit()
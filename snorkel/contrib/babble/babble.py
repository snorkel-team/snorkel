"""
add Babbler object
    builds with candidate set, builds semantic parser internally
    user requests a candidate, views candidate, writes explanation
    user inputs an explanation (which gets tied to that candidate
    on request, prints past explanations given
    outputs candidates that have no labels yet 
        (after that, candidates w/ most conflict/least support?)
when explanation is received:
	convert to LF(s)
	apply to dev set
	filter uniform/duplicate
	report precision, recall, make (tp, fp, tn, fn) available
	possibly let them select the better interpretation
when done, pull out L_train and proceed to generative model
"""
from collections import defaultdict, namedtuple

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sparse
import random

from snorkel.annotations import LabelAnnotator, load_gold_labels, csr_AnnotationMatrix
from snorkel.learning.utils import MentionScorer
from snorkel.lf_helpers import test_LF
from snorkel.utils import matrix_tp, matrix_fp, matrix_tn, matrix_fn, matrix_coverage

from snorkel.contrib.babble.filter_bank import FilterBank
from snorkel.contrib.babble.grammar import Parse
from snorkel.contrib.babble.semparser import Explanation, SemanticParser

# from tutorials.babble.spouse.spouse_examples import get_user_lists, get_explanations

ConfusionMatrix = namedtuple('ConfusionMatrix', ['tp', 'fp', 'tn', 'fn'])
Statistics = namedtuple('Statistics', ['accuracy', 'class_coverage'])

class CandidateGenerator(object):
    """
    A generator for returning a list of candidates in a certain order.
    """
    def __init__(self, candidates, strategy='linear'):
        if strategy == 'linear':
            self.candidate_generator = self.linear_generator(candidates)
        elif strategy == 'random':
            self.candidate_generator = self.random_generator(candidates)
        elif strategy == 'balanced':
            raise NotImplementedError
        elif strategy == 'active':
            raise NotImplementedError
        else:
            raise Exception("kwarg 'strategy' must be in "
                "{'linear', 'random', 'balanced', 'active'}")

    def __iter__(self):
        return self

    def next(self):
        return self.candidate_generator.next()

    def linear_generator(self, candidates):
        for c in candidates:
            yield c

    def random_generator(self, candidates):
        random.shuffle(candidates)
        for c in candidates:
            yield c


class BabbleStream(object):
    """
    An object for iteratively viewing candidates and parsing corresponding explanations.
    """
    def __init__(self, session, mode='text', candidate_class=None, 
                strategy='linear', verbose=True):
        self.session = session
        self.mode = mode
        self.candidate_class = candidate_class
        self.verbose = verbose

        self.dev_candidates = session.query(self.candidate_class).filter(self.candidate_class.split == 1).all()
        self.candidate_generator = CandidateGenerator(self.dev_candidates, strategy)
        self.user_lists = {}
        self.semparser = None
        self.filter_bank = FilterBank(session, candidate_class)
        
        self.explanations = set()
        self.parses = set()
        self.label_matrix = None

        # Temporary storage
        self.temp_explanations = None
        self.temp_parses = None
        self.temp_label_matrix = None

        # Evaluation tools
        dev_labels          = load_gold_labels(session, annotator_name='gold', split=1)
        self.num_dev_total  = len(self.dev_candidates)
        self.num_dev_pos    = dev_labels.nnz
        self.num_dev_neg    = self.num_dev_total - self.num_dev_pos
        self.scorer         = MentionScorer(self.dev_candidates, dev_labels)


    def __iter__(self):
        return self

    def next(self):
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
            parses, _, _ = self.apply(explanations)
            if parses:
                self.commit()

    def apply(self, explanations):
        """
        :param explanations: an Explanation or list of Explanations.
        """
        if self.temp_parses:
            self.commit([])
            print("All previously uncommitted parses have been flushed.")

        parses = self._parse(explanations)
        parses, label_matrix = self._filter(parses, explanations)
        conf_matrix_list, stats_list = self.analyze(parses)
        
        # Hold results in temporary space until commit
        self.temp_explanations = explanations if isinstance(explanations, list) else [explanations]
        self.temp_parses = parses if isinstance(parses, list) else [parses]
        self.temp_label_matrix = label_matrix
        
        return parses, conf_matrix_list, stats_list

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
        :return: a list of Parses
        :return: a sparse.csr_matrix with shape [num_candidates, len(parses)]
        """
        # Filter
        parses, label_matrix = self.filter_bank.apply(parses, explanations)

        return parses, label_matrix

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

            conf_matrix_list.append(ConfusionMatrix(tp, fp, tn, fn))
            TP, FP, TN, FN = map(lambda x: float(len(x)), [tp, fp, tn, fn])
            if TP or FP:
                accuracy = TP/float(TP + FP)
                class_coverage = float(TP + FP)/self.num_dev_pos
            elif TN or FN:
                accuracy = TN/float(TN + FN)
                class_coverage = float(TN + FN)/self.num_dev_pos
            else:
                accuracy = None
                class_coverage = None
            stats_list.append(Statistics(accuracy, class_coverage))

        return conf_matrix_list, stats_list

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

            parses_to_add = set(p for i, p in enumerate(self.temp_parses) if i in idxs)
            parse_names_to_add = [p.function.__name__ for p in parses_to_add]
            explanations_to_add = set(e for e in self.temp_explanations if 
                any(pn.startswith(e.name) for pn in parse_names_to_add))
      
            self.parses.update(parses_to_add)
            self.explanations.update(explanations_to_add)
            if self.label_matrix is None:
                self.label_matrix = self.temp_label_matrix
            else:
                self.label_matrix = sparse.hstack((self.label_matrix, self.temp_label_matrix))

            if self.verbose:
                print("Added {} parse(s) to set. (Total # parses = {})".format(
                    len(parses_to_add), len(self.parses)))
                print("Added {} explanation(s) to set. (Total # explanations = {})".format(
                    len(explanations_to_add), len(self.parses)))
        
        # Permanently store the semantics and signatures in duplicate filters
        self.filter_bank.commit(idxs)

        self.temp_parses = None
        self.temp_explanations = None
        self.temp_label_matrix = None

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



class Babbler(object):
    # TODO: convert to UDFRunner 
    def __init__(self, mode, candidate_class=None, explanations=[], exp_names=[], 
                 user_lists={}, string_format='implicit', beam_width=10, top_k=-1,
                 do_filter_duplicate_semantics=True, 
                 do_filter_consistency=True, 
                 do_filter_duplicate_signatures=True, 
                 do_filter_uniform_signatures=True,
                 do_filter_low_accuracy=False, acc_threshold=0.55, gold_labels=None,
                 verbose=True):
        self.candidate_class = candidate_class
        self.user_lists = user_lists
        self.semparser = SemanticParser(
            mode=mode, candidate_class=candidate_class, user_lists=user_lists,
            string_format=string_format, beam_width=beam_width, top_k=top_k)
        self.semparser.name_explanations(explanations, exp_names)
        if len(explanations) != len(set([exp.name for exp in explanations])):
            raise Exception("All Explanations must have unique names.")
        self.explanations = explanations
        self.explanations_by_name = {}
        self.update_explanation_map(explanations)
        self.do_filter_duplicate_semantics = do_filter_duplicate_semantics
        self.do_filter_consistency = do_filter_consistency
        self.do_filter_duplicate_signatures = do_filter_duplicate_signatures,
        self.do_filter_uniform_signatures = do_filter_uniform_signatures
        self.do_filter_low_accuracy = do_filter_low_accuracy
        self.gold_labels = gold_labels
        self.acc_threshold = acc_threshold
        self.verbose = verbose
        self.lfs = []
        self.label_matrix = None
        self.labeler = None
    
    def add_explanations(self, new_explanations):
        new_explanations = (new_explanations if isinstance(new_explanations, list) 
                            else [new_explanations])
        if not isinstance(new_explanations[0], Explanation):
            raise Exception("Argument to add_explanations() must be an Explanation "
                "object or list of Explanations.")
        self.explanations.extend(new_explanations)
        self.update_explanation_map(new_explanations)

    def update_explanation_map(self, explanations):
        for exp in explanations:
            self.explanations_by_name[exp.name] = exp
    
    def add_user_lists(self, new_lists):
        new_lists = new_lists if isinstance(new_lists, list) else [new_lists]
        if not isinstance(new_lists[0], dict):
            raise Exception("Argument to add_user_lists() must be a dictionary "
                "with (list_name: list_values) items.")
        for user_list in new_lists:
            for k, v in user_list.items():
                self.user_lists[k] = v

    def generate_lfs(self):
        """Converts explanations into LFs."""
        if not self.explanations:
            raise Exception("Could not find explanations.")
        self.parses = self.semparser.parse(self.explanations, return_parses=True, verbose=self.verbose)
        self.lfs = [parse.function for parse in self.parses]
        print("Parsed {} LFs from {} explanations.".format(
            len(self.lfs), len(self.explanations)))
        return self.lfs

    def filter_duplicate_semantics(self):
        """Filters out LFs with identical logical forms (keeping one)."""
        seen = set()
        duplicates = []
        non_duplicates = []
        for parse in self.parses:
            if hash(parse.semantics) not in seen:
                non_duplicates.append(parse)
                seen.add(hash(parse.semantics))
            else:
                duplicates.append(parse)
        self.parses = non_duplicates
        self.lfs = [parse.function for parse in self.parses]
        print("Filtered to {} LFs with duplicate semantics filter ({} filtered).".format(
            len(non_duplicates), len(duplicates)))

    def filter_consistency(self):
        """Filters out LFs that incorrectly label their accompanying candidate."""
        if not self.lfs:
            raise Exception("Could not find lfs.")
        explanation_dict = {}
        for exp in self.explanations:
            if exp.candidate and not isinstance(exp.candidate, self.candidate_class):
                pass
                # raise TypeError("Expected type {}, got {} for candidate {}.".format(
                #     self.candidate_class, type(exp.candidate), exp.candidate))
            explanation_dict[exp.name] = exp
        consistent = []
        inconsistent = []
        unknown = []
        for parse in self.parses:
            lf = parse.function
            exp_name = extract_exp_name(lf)
            exp = explanation_dict[exp_name]
            if isinstance(exp.candidate, self.candidate_class):
                if lf(exp.candidate):
                    consistent.append(parse)
                else:
                    inconsistent.append(parse)
            else:
                unknown.append(parse)
        if unknown:
            print("Note: {} LFs did not have candidates and therefore could "
                  "not be filtered.".format(len(unknown)))
        print("Filtered to {} LFs with consistency filter ({} filtered).".format(
            len(consistent) + len(unknown), len(inconsistent)))
        self.parses = consistent + unknown
        self.lfs = [parse.function for parse in self.parses]

    def generate_label_matrix(self, split=0, parallelism=1):
        if not self.lfs:
            raise Exception("Could not find lfs.")
        self.labeler = LabelAnnotator(lfs=self.lfs)
        self.label_matrix = self.labeler.apply(split=split, parallelism=parallelism)
        return self.label_matrix

    def filter_uniform_signatures(self):
        """Filters out LFs with uniform labeling signatures."""
        if self.label_matrix is None:
            raise Exception("Could not find label_matrix.")
        non_uniform = []
        num_lfs = self.label_matrix.shape[1]
        for i in range(num_lfs):
            if abs(np.sum(self.label_matrix[:,i])) not in [0, self.label_matrix.shape[0]]:
                non_uniform.append(i)
        self.label_matrix = self.label_matrix[:, non_uniform]
        self.parses = [parse for i, parse in enumerate(self.parses) if i in set(non_uniform)]
        self.lfs = [parse.function for parse in self.parses]
        print("Filtered to {} LFs with uniform signatures filter ({} filtered).".format(
            len(non_uniform), num_lfs - len(non_uniform)))

    def filter_duplicate_signatures(self):
        """Filters out LFs with identical labeling signatures (keeping one)."""
        if self.label_matrix is None:
            raise Exception("Could not find label_matrix.")
        duplicate_hashes = set([])
        non_duplicates = []
        num_lfs = self.label_matrix.shape[1]
        for i in range(num_lfs):
            h = hash(self.label_matrix[:,i].nonzero()[0].tostring())
            if h not in duplicate_hashes:
                non_duplicates.append(i)
                duplicate_hashes.add(h)
        self.label_matrix = self.label_matrix[:, non_duplicates]
        self.parses = [parse for i, parse in enumerate(self.parses) if i in set(non_duplicates)]
        self.lfs = [parse.function for parse in self.parses]
        print("Filtered to {} LFs with duplicate signatures filter ({} filtered).".format(
            len(non_duplicates), num_lfs - len(non_duplicates)))                

    def filter_low_accuracy(self):
        """Filters out LFs with accuracy on gold data less than self.acc_threshold."""
        if self.label_matrix is None:
            raise Exception("Could not find label_matrix.")
        if self.gold_labels is None:
            raise Exception("Could not find gold_labels.")
        labels = self.gold_labels
        ls = np.ravel(labels.todense() if sparse.issparse(labels) else labels)
        tp = matrix_tp(self.label_matrix, ls)
        fp = matrix_fp(self.label_matrix, ls)
        tn = matrix_tn(self.label_matrix, ls)
        fn = matrix_fn(self.label_matrix, ls)
        ac = (tp+tn).astype(float) / (tp+tn+fp+fn)
        low_accuracy = []
        high_accuracy = []
        num_lfs = self.label_matrix.shape[1]
        for i, accuracy in enumerate(ac):
            if accuracy < self.acc_threshold:
                low_accuracy.append(i)
            else:
                high_accuracy.append(i)
        self.label_matrix = self.label_matrix[:, high_accuracy]
        self.parses = [parse for i, parse in enumerate(self.parses) if i in set(high_accuracy)]
        self.lfs = [parse.function for parse in self.parses]
        print("Filtered to {} LFs with low accuracy filter ({} filtered).".format(
            len(high_accuracy), num_lfs - len(high_accuracy)))
        

    def apply(self, split=0, parallelism=1):
        """Applies entire Babble Labble pipeline: convert, label, filter."""
        self.generate_lfs()
        if self.do_filter_duplicate_semantics:
            self.filter_duplicate_semantics()
        if self.do_filter_consistency: 
            self.filter_consistency()
        self.generate_label_matrix(split=split, parallelism=parallelism)
        if self.do_filter_uniform_signatures:
            self.filter_uniform_signatures()
        if self.do_filter_duplicate_signatures:
            self.filter_duplicate_signatures()
        if self.do_filter_low_accuracy:
            self.filter_low_accuracy()
        return self.label_matrix

    def get_explanations(self):
        exp_names = []
        for lf in self.lfs:
            exp_names.append(extract_exp_name(lf))
        return sorted([self.explanations_by_name[exp_name] for exp_name in exp_names],
            key=lambda x: x.name)

    def get_parses(self, semantics=True, translate=True):
        parses = sorted(self.parses, key=lambda x: extract_exp_name(x.function))
        if semantics:
            semantics = [p.semantics for p in parses]
            if translate:
                return [self.translate(s) for s in semantics]
            else:
                return semantics
        else:
            return parses

    def get_lfs(self):
        return [parse.function for parse in self.get_parses(semantics=False)]

    def translate(self, semantics):
        return self.semparser.translate(semantics)

    def display_lf_distribution(self):
        def count_parses_by_exp(lfs):
            num_parses_by_exp = defaultdict(int)
            for lf in lfs:
                exp_name = extract_exp_name(lf)
                num_parses_by_exp[exp_name] += 1
            return num_parses_by_exp.values()

        # print("Total Explanations: {}".format(len(explanations)))
        # print("Total parse-able Explanations: {}".format(len(num_parses_by_exp)))
        num_parses_by_exp = count_parses_by_exp(self.lfs)
        print("{} LFs from {} out of {} explanation(s)".format(
            len(self.lfs), len(self.explanations) - num_parses_by_exp.count(0), 
            len(self.explanations)))
        plt.hist(num_parses_by_exp, 
            bins=range(max(num_parses_by_exp) + 2), align='left')
        plt.xticks(range(max(num_parses_by_exp) + 2))
        plt.xlabel("# of LFs")
        plt.ylabel("# of Explanations")
        plt.title('# LFs per Explanation')
        plt.show()

def extract_exp_name(lf):
    return lf.__name__[:lf.__name__.rindex('_')]

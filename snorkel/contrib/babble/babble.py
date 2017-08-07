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
import collections

import matplotlib.pyplot as plt
import numpy as np

from semparser import Explanation, SemanticParser
from snorkel.annotations import LabelAnnotator

class Babbler(object):
    # TODO: convert to UDFRunner 
    def __init__(self, candidate_class, explanations=[], exp_names=[], user_lists={}, 
                 do_filter_duplicate_semantics=True, 
                 do_filter_consistency=True, 
                 do_filter_duplicate_signatures=True, 
                 do_filter_uniform_signatures=True,
                 verbose=True):
        self.candidate_class = candidate_class
        self.user_lists = user_lists
        self.semparser = SemanticParser(candidate_class=candidate_class, user_lists=user_lists)
        self.semparser.name_explanations(explanations, exp_names)
        if len(explanations) != len(set([exp.name for exp in explanations])):
            raise Exception("All Explanations must have unique names.")
        self.explanations = explanations
        self.explanations_by_name = {}
        self.update_explanation_map(explanations)
        self.do_filter_duplicate_semantics = do_filter_duplicate_semantics, 
        self.do_filter_consistency = do_filter_consistency, 
        self.do_filter_duplicate_signatures = do_filter_duplicate_signatures, 
        self.do_filter_uniform_signatures = do_filter_uniform_signatures,
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
        # print("Parsed {} LFs from {} explanations.".format(
        #     len(self.lfs), len(self.explanations)))
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
                raise TypeError("Expected type {}, got {} for candidate {}.".format(
                    self.candidate_class, type(exp.candidate), candidate))
            explanation_dict[exp.name] = exp
        consistent = []
        inconsistent = []
        unknown = []
        for parse in self.parses:
            lf = parse.function
            exp_name = extract_exp_name(lf)
            exp = explanation_dict[exp_name]
            if exp.candidate:
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

    def load_matrix(self, session, split=0):
        if self.labeler is None:
            self.labeler = LabelAnnotator(lfs=self.lfs)
        self.label_matrix = self.labeler.load_matrix(session, split=split)
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
        return self.label_matrix

    def get_explanations(self):
        exp_names = []
        for lf in self.lfs:
            exp_names.append(extract_exp_name(lf))
        return sorted([self.explanations_by_name[exp_name] for exp_name in exp_names],
            key=lambda x: x.name)

    def get_parses(self):
        return sorted(self.parses, key=lambda x: extract_exp_name(x.function))

    def get_lfs(self):
        return [parse.function for parse in self.get_parses()]

    def display_lf_distribution(self):
        def count_parses_by_exp(lfs):
            num_parses_by_exp = collections.defaultdict(int)
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
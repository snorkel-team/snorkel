from collections import namedtuple, OrderedDict
import numpy as np

from scipy.sparse import csr_matrix

from snorkel.annotations import LabelAnnotator, csr_AnnotationMatrix
from snorkel.utils import ProgressBar, PrintTimer
from snorkel.contrib.babble.grammar import Parse

FilteredParse = namedtuple('FilteredParse', ['parse', 'reason'])

class FilterBank(object):
    def __init__(self, session, candidate_class=None, split=1):
        self.session = session
        self.candidate_class = candidate_class
        self.split = split
        self.dup_semantics_filter = DuplicateSemanticsFilter()
        self.consistency_filter = ConsistencyFilter(self.candidate_class)
        self.uniform_filter = UniformSignatureFilter()
        self.dup_signature_filter = DuplicateSignatureFilter()
        self.label_matrix = None

    def apply(self, parses, explanations, parallelism=1):
        """
        Returns:
            parses: Parses
            filtered_parses: dict of Parses removed by each Filter
            label_matrix: csr_AnnotationMatrix corresponding to parses
        """
        filtered_parses = {}

        # Apply structure and consistency based filters
        parses, rejected = self.dup_semantics_filter.filter(parses)
        filtered_parses[self.dup_semantics_filter.name()] = rejected
        if not parses: return parses, filtered_parses, None

        parses, rejected = self.consistency_filter.filter(parses, explanations)
        filtered_parses[self.consistency_filter.name()] = rejected
        if not parses: return parses, filtered_parses, None

        # Label and extract signatures
        label_matrix = self.label(parses)

        # Apply signature based filters
        parses, rejected, label_matrix = self.uniform_filter.filter(parses, label_matrix)
        filtered_parses[self.uniform_filter.name()] = rejected
        if not parses: return parses, filtered_parses, None

        parses, rejected, label_matrix = self.dup_signature_filter.filter(parses, label_matrix)
        filtered_parses[self.dup_signature_filter.name()] = rejected
        if not parses: return parses, filtered_parses, None

        return parses, filtered_parses, label_matrix

    def label(self, parses):
        # TODO: replace this naive double for loop with a parallelized solution
        with PrintTimer("Applying labeling functions to split {}".format(self.split)):
            lfs = [parse.function for parse in parses]
            candidates = self.session.query(self.candidate_class).filter(
                self.candidate_class.split == self.split).all()
            dense_label_matrix = np.zeros((len(candidates), len(lfs)))

            pb = ProgressBar(len(lfs))
            for j, lf in enumerate(lfs):
                pb.bar(j)
                for i, c in enumerate(candidates):
                    dense_label_matrix[i, j] = lf(c)
            pb.close()                
            label_matrix = csr_matrix(dense_label_matrix)
            return label_matrix

    def commit(self, idxs):
        self.dup_semantics_filter.commit(idxs)
        self.consistency_filter.commit(idxs)
        self.uniform_filter.commit(idxs)
        self.dup_signature_filter.commit(idxs)


class Filter(object):
    # def __init__(self)
    def filter(self, input):
        raise NotImplementedError
    
    def name(self):
        return type(self).__name__

    def commit(self, idxs):
        pass

    def validate(self, parses):
        parses = parses if isinstance(parses, list) else [parses]
        if not parses:
            print("Warning: Filter {} was applied to an empty list.".format(self.name()))
        if parses and not isinstance(parses[0], Parse):
            raise ValueError("Expected: Parse. Got: {}".format(type(parses[0])))
        return parses


class DuplicateSemanticsFilter(Filter):
    """Filters out parses with identical logical forms (keeping one)."""
    def __init__(self):
        self.seen_semantics = {}        # key: semantics, value: parses
        self.temp_seen_semantics = OrderedDict()   # key: semantics, value: parses
    
    def filter(self, parses):
        parses = self.validate(parses)
        if not parses: return [], []

        good_parses = []
        bad_parses = []
        for parse in parses:
            # If a parse collides with a previously committed parse or a newly
            # seen temporary parse, add it to the bad parses and store the parse
            # that it collided with, for reference. Otherwise, add to good parses.
            if parse.semantics in self.seen_semantics:
                bad_parses.append(FilteredParse(parse, self.seen_semantics[parse.semantics]))
            elif parse.semantics in self.temp_seen_semantics:
                # Store the removed parse, and the parse it collided with, for reference.
                bad_parses.append(FilteredParse(parse, self.temp_seen_semantics[parse.semantics]))
            else:
                good_parses.append(parse)
                self.temp_seen_semantics[parse.semantics] = parse

        print("{} parse(s) remain ({} parse(s) removed by {}).".format(
            len(good_parses), len(bad_parses), self.name()))    
        return good_parses, bad_parses

    def commit(self, idxs):
        for i, (semantics, parse) in enumerate(self.temp_seen_semantics.items()):
             if i in idxs:
                 self.seen_semantics[parse.semantics] =  parse
        self.temp_seen_semantics = OrderedDict()


class ConsistencyFilter(Filter):
    """Filters out parses that incorrectly label their accompanying candidate."""
    def __init__(self, candidate_class):
        self.candidate_class = candidate_class

    def filter(self, parses, explanations):
        """
        If possible, confirm that candidate is of proper type.
        If candidate_class was not provided, use try/except instead.
        """
        parses = self.validate(parses)
        if not parses: return [], []

        explanations = explanations if isinstance(explanations, list) else [explanations]
        explanation_dict = {exp.name: exp for exp in explanations}
        
        good_parses = []
        bad_parses = []
        unknown_parses = []
        for parse in parses:
            lf = parse.function
            exp_name = extract_exp_name(lf)
            exp = explanation_dict[exp_name]
            if self.candidate_class:
                if isinstance(exp.candidate, self.candidate_class):
                    try:
                        if lf(exp.candidate):
                            good_parses.append(parse)
                        else:
                            bad_parses.append(FilteredParse(parse, exp.candidate))
                    except UnicodeDecodeError:
                        import pdb; pdb.set_trace()
                        print("Warning: skipped consistency evaluation because of UnicodeDecode error.")
                else:
                    unknown_parses.append(parse)
            else:
                try:
                    if lf(exp.candidate):
                        good_parses.append(parse)
                    else:
                        bad_parses.append(FilteredParse(parse, exp.candidate))
                except:
                    unknown_parses.append(parse)
        if unknown_parses:
            print("Note: {} LFs did not have candidates and therefore could "
                  "not be filtered.".format(len(unknown_parses)))
    
        good_parses += unknown_parses
        print("{} parse(s) remain ({} parse(s) removed by {}).".format(
            len(good_parses), len(bad_parses), self.name())) 
        return good_parses, bad_parses


class UniformSignatureFilter(Filter):
    """Filters out parses that give all candidates the same label."""
    def filter(self, parses, label_matrix):
        """
        :param parses: ...
        :param label_matrix: a csr_sparse matrix of shape [M, N] for M 
            candidates by the N Parses in :param parses.
        :returns good_parses: a list of Parses to keep
        :returns bad_parses: a list of Parses to not keep
        :returns label_matrix: a csr_sparse matrix of shape [M, N'] for M 
            candidates by the N' Parses in good_parses.
        """
        parses = self.validate(parses)
        if not parses: return [], [], label_matrix
        if not isinstance(label_matrix, csr_matrix):
            raise Exception("Method filter() requires a label_matrix of type "
                "scipy.sparse.csr_matrix.")

        num_candidates, num_lfs = label_matrix.shape
        column_sums = np.asarray(abs(np.sum(label_matrix, 0))).ravel()
        labeled_none_idxs = [i for i, sum in enumerate(column_sums) if sum == 0]
        labeled_all_idxs = [i for i, sum in enumerate(column_sums) if sum == num_candidates]
        uniform_idxs = labeled_none_idxs + labeled_all_idxs
        nonuniform_idxs = [i for i, sum in enumerate(column_sums) if i not in uniform_idxs]

        good_parses = [parse for i, parse in enumerate(parses) if i in nonuniform_idxs]
        bad_parses = []
        for i, parse in enumerate(parses):
            if i in labeled_all_idxs:
                bad_parses.append(FilteredParse(parse, "ALL"))
            elif i in labeled_none_idxs:
                bad_parses.append(FilteredParse(parse, "NONE"))
            
        label_matrix = label_matrix[:, nonuniform_idxs]

        print("{} parse(s) remain ({} parse(s) removed by {}: ({} None, {} All)).".format(
            len(good_parses), len(bad_parses), self.name(), 
            len(labeled_none_idxs), len(labeled_all_idxs)))

        return good_parses, bad_parses, label_matrix


class DuplicateSignatureFilter(Filter):
    """Filters out all but one parse that have the same labeling signature."""
    def __init__(self):
        self.seen_signatures = {}        # key: signature hash, value: parse
        self.temp_seen_signatures = OrderedDict()   # key: signature hash, value: parse

    def filter(self, parses, label_matrix):
        """
        :param parses: ...
        :param label_matrix: a label_matrix corresponding to only the remaining 
            parses from this batch.
        """
        parses = self.validate(parses)
        if not parses: return [], [], label_matrix
        if not isinstance(label_matrix, csr_matrix):
            raise Exception("Method filter() requires a label_matrix of type "
                "scipy.sparse.csr_matrix.")    

        num_candidates, num_lfs = label_matrix.shape
        signatures = [hash(label_matrix[:,i].nonzero()[0].tostring()) for i in range(num_lfs)]
        good_parses = []
        bad_parses = []
        nonduplicate_idxs = []
        for i, (sig, parse) in enumerate(zip(signatures, parses)):
            # If a parse collides with a previously committed parse or a newly
            # seen temporary parse, add it to the bad parses and store the parse
            # that it collided with, for reference. Otherwise, add to good parses.
            if sig in self.seen_signatures:
                bad_parses.append(FilteredParse(parse, self.seen_signatures[sig]))
            elif sig in self.temp_seen_signatures:
                bad_parses.append(FilteredParse(parse, self.temp_seen_signatures[sig]))
            else:
                good_parses.append(parse)
                self.temp_seen_signatures[sig] = parse
                nonduplicate_idxs.append(i)

        label_matrix = label_matrix[:, nonduplicate_idxs]

        print("{} parse(s) remain ({} parse(s) removed by {}).".format(
            len(good_parses), len(bad_parses), self.name()))    
        return good_parses, bad_parses, label_matrix
    
    def commit(self, idxs):
        for i, (sig, parse) in enumerate(self.temp_seen_signatures.items()):
             if i in idxs:
                 self.seen_signatures[sig] =  parse
        self.temp_seen_signatures= OrderedDict()


def extract_exp_name(lf):
    return lf.__name__[:lf.__name__.rindex('_')]
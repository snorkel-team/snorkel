import numpy as np

from snorkel.annotations import LabelAnnotator, csr_AnnotationMatrix
from snorkel.contrib.babble.grammar import Parse

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

    # TODO: make a UDFRunner
    def apply(self, parses, explanations, parallelism=1):
        """
        Returns:
            parses: Parses
            label_matrix: csr_AnnotationMatrix corresponding to parses
        """
        # Apply structure and consistency based filters
        parses, rejected = self.dup_semantics_filter.filter(parses)
        if not parses: return [], None

        parses, rejected = self.consistency_filter.filter(parses, explanations)
        if not parses: return [], None

        # Label and extract signatures
        lfs = [parse.function for parse in parses]
        labeler = LabelAnnotator(lfs=lfs)

        if self.label_matrix is None:
            label_matrix = labeler.apply(split=self.split, parallelism=parallelism)
        else:
            label_matrix = labeler.apply_existing(split=self.split, parallelism=parallelism)

        # Apply signature based filters
        parses, rejected, label_matrix = self.uniform_filter.filter(parses, label_matrix)
        label_matrix = label_matrix
        if not parses: return [], label_matrix

        parses, rejected, label_matrix = self.dup_signature_filter.filter(parses, label_matrix)
        label_matrix = label_matrix
        if not parses: return [], label_matrix

        self.label_matrix = label_matrix 
        return parses, label_matrix

        # for col_idx in range(self.label_matrix.shape[1]):
        #     print(self.label_matrix.get_key(self.session, 0).name)

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
        self.seen = set()
        self.temp_seen = []
    
    def filter(self, parses):
        self.validate(parses)
        if not parses: return [], []

        good_parses = []
        bad_parses = []
        for parse in parses:
            h = hash(parse.semantics) 
            if h in self.seen or h in self.temp_seen:
                bad_parses.append(parse)
            else:
                good_parses.append(parse)
                self.temp_seen.append(h)

        print("{} parse(s) remain ({} parse(s) removed by {}).".format(
            len(good_parses), len(bad_parses), self.name()))    
        return good_parses, bad_parses

    def commit(self, idxs):
        self.seen.update([h for i, h in enumerate(self.temp_seen) if i in idxs])
        self.temp_seen = []


class ConsistencyFilter(Filter):
    """Filters out parses that incorrectly label their accompanying candidate."""
    def __init__(self, candidate_class):
        self.candidate_class = candidate_class

    def filter(self, parses, explanations):
        """
        If possible, confirm that candidate is of proper type.
        If candidate_class was not provided, use try/except instead.
        """
        self.validate(parses)
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
                    if lf(exp.candidate):
                        good_parses.append(parse)
                    else:
                        bad_parses.append(parse)
                else:
                    unknown_parses.append(parse)
            else:
                try:
                    if lf(exp.candidate):
                        good_parses.append(parse)
                    else:
                        bad_parses.append(parse)
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
        self.validate(parses)
        if not parses: return [], [], label_matrix
        if not isinstance(label_matrix, csr_AnnotationMatrix):
            raise Exception("Method filter() requires a label_matrix of type "
                "csr_AnnotationMatrix.")

        num_candidates, num_lfs = label_matrix.shape
        column_sums = np.asarray(abs(np.sum(label_matrix, 0))).ravel()
        nonuniform_idxs = [i for i, sum in enumerate(column_sums) if sum not in (0, num_candidates)]

        good_parses = [parse for i, parse in enumerate(parses) if i in list(nonuniform_idxs)]
        bad_parses = [parse for i, parse in enumerate(parses) if i not in list(nonuniform_idxs)]
        label_matrix = label_matrix[:, nonuniform_idxs]

        print("{} parse(s) remain ({} parse(s) removed by {}).".format(
            len(good_parses), len(bad_parses), self.name()))    
        return good_parses, bad_parses, label_matrix


class DuplicateSignatureFilter(Filter):
    """Filters out all but one parse that have the same labeling signature."""
    def __init__(self):
        self.seen = set()
        self.temp_seen = []

    def filter(self, parses, label_matrix):
        self.validate(parses)
        if not parses: return [], [], label_matrix
        if not isinstance(label_matrix, csr_AnnotationMatrix):
            raise Exception("Method filter() requires a label_matrix of type "
                "csr_AnnotationMatrix.")    

        num_candidates, num_lfs = label_matrix.shape
        signatures = [hash(label_matrix[:,i].nonzero()[0].tostring()) for i in range(num_lfs)]
        nonduplicate_idxs = []
        for i, sig in enumerate(signatures):
            if sig not in self.seen and sig not in self.temp_seen:
                nonduplicate_idxs.append(i)
            self.temp_seen.append(sig)

        good_parses = [parse for i, parse in enumerate(parses) if i in list(nonduplicate_idxs)]
        bad_parses = [parse for i, parse in enumerate(parses) if i not in list(nonduplicate_idxs)]        
        label_matrix = label_matrix[:, nonduplicate_idxs]

        print("{} parse(s) remain ({} parse(s) removed by {}).".format(
            len(good_parses), len(bad_parses), self.name()))    
        return good_parses, bad_parses, label_matrix
    
    def commit(self, idxs):
        self.seen.update([h for i, h in enumerate(self.temp_seen) if i in idxs])
        self.temp_seen = []


def extract_exp_name(lf):
    return lf.__name__[:lf.__name__.rindex('_')]
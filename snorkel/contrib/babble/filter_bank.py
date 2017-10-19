from snorkel.annotations import LabelAnnotator
from snorkel.contrib.babble.grammar import Parse

class FilterBank(object):
    def __init__(self, session, candidate_class=None, split=1):
        self.session = session
        self.candidate_class = candidate_class
        self.split = split
        self.duplicate_semantics = DuplicateSemanticsFilter()
        self.consistency = ConsistencyFilter(self.candidate_class)
        # self.f2 = UniformSignatureFilter(???)
        # self.f3 = DuplicateSignatureFilter(???)
        self.label_matrix = None

    # TODO: make a UDFRunner
    def apply(self, parses, explanations, parallelism=1):
        
        # PLAN:
        """
        apply filters that use parses
        use LabelAnnotator to get label matrix
        extract signatures
        apply filters that use signatures
        """
        # Apply structure and consistency based filters
        parses, _ = self.duplicate_semantics.filter(parses)
        if not parses: return [], [], []

        parses, _ = self.consistency.filter(parses, explanations)
        if not parses: return [], [], []

        # Label and extract signatures
        # lfs = [parse.function for parse in parses]
        # labeler = LabelAnnotator(lfs=lfs)
        # if not self.label_matrix:
        #     self.label_matrix = labeler.apply(split=self.split, parallelism=parallelism)
        # else:
        #     self.label_matrix = labeler.apply_existing(split=self.split, parallelism=parallelism)

        # TODO: complete this code for pulling out signatures from label matrix
        # for col_idx in range(self.label_matrix.shape[1]):
        #     print(self.label_matrix.get_key(self.session, 0).name)
        
        return parses, None

        # self.generate_lfs()
        # if self.do_filter_duplicate_semantics:
        #     self.filter_duplicate_semantics()
        # if self.do_filter_consistency: 
        #     self.filter_consistency()
        # self.generate_label_matrix(split=split, parallelism=parallelism)
        # if self.do_filter_uniform_signatures:
        #     self.filter_uniform_signatures()
        # if self.do_filter_duplicate_signatures:
        #     self.filter_duplicate_signatures()
        # if self.do_filter_low_accuracy:
        #     self.filter_low_accuracy()
        # return self.label_matrix

class Filter(object):
    # def __init__(self)
    def filter(self, input):
        raise NotImplementedError
    
    def name(self):
        return type(self).__name__

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
    
    def filter(self, parses):
        self.validate(parses)
        if not parses: return [], []

        good_parses = []
        bad_parses = []
        for parse in parses:
            if hash(parse.semantics) in self.seen:
                bad_parses.append(parse)
            else:
                good_parses.append(parse)
                self.seen.add(hash(parse.semantics))

        print("{} parse(s) remain ({} parse(s) removed by {}).".format(
            len(good_parses), len(bad_parses), self.name()))    
        return good_parses, bad_parses


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


def extract_exp_name(lf):
    return lf.__name__[:lf.__name__.rindex('_')]
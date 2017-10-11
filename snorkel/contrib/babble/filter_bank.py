from snorkel.contrib.babble.grammar import Parse

class FilterBank(object):
    def __init__(self, candidate_class=None, split=1):
        self.candidate_class = candidate_class
        self.split = split

    # TODO: make a UDFRunner
    def apply(self, parses, explanations, parallelism=1):
        f0 = DuplicateSemanticsFilter()
        f1 = ConsistencyFilter(self.candidate_class, explanations)
        # f2 = UniformSignatureFilter()
        # f3 = DuplicateSignatureFilter()
        
        parses, _ = f0.filter(parses)
        parses, _ = f1.filter(parses)

        return parses, None
        # lfs = [parse.function for parse in parses]
        # labeler = LabelAnnotator(lfs=lfs)
        # # TODO: use apply_existing
        # self.label_matrix = self.labeler.apply(split=self.split, parallelism=parallelism)


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

class ParseFilter(Filter):
    def filter(self, parses):
        parses = parses if isinstance(parses, list) else [parses]
        if not parses:
            raise Warning("Filter {} was applied to an empty list.".format(self.name()))
        if not isinstance(parses[0], Parse):
            raise ValueError("Expected: Parse. Got: {}".format(type(parses[0])))
        return self._filter(parses)

class DuplicateSemanticsFilter(ParseFilter):
    """Filters out parses with identical logical forms (keeping one)."""
    def __init__(self):
        self.seen = set()
    
    def _filter(self, parses):
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

class ConsistencyFilter(ParseFilter):
    """Filters out parses that incorrectly label their accompanying candidate."""
    def __init__(self, candidate_class, explanations):
        self.candidate_class = candidate_class
        explanations = explanations if isinstance(explanations, list) else [explanations]
        self.explanation_dict = {exp.name: exp for exp in explanations}

    def filter(self, parses):
        """
        If possible, confirm that candidate is of proper type.
        If candidate_class was not provided, use try/except isntead.
        """
        good_parses = []
        bad_parses = []
        unknown_parses = []
        for parse in parses:
            lf = parse.function
            exp_name = extract_exp_name(lf)
            exp = self.explanation_dict[exp_name]
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
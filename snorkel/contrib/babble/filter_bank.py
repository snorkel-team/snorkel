from snorkel.contrib.babble.grammar import Parse

class FilterBank(object):
    def __init__(self, split=1):
        self.split = split

    # TODO: make a UDFRunner
    def apply(self, parses, parallelism=1):
        f0 = DuplicateSemanticsFilter()
        # f1 = ConsistencyFilter()
        # f2 = UniformSignatureFilter()
        # f3 = DuplicateSignatureFilter()
        
        parses, _ = f0.filter(parses)
        # parses = f1.filter(parses)

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
        print("Filter {} removed {} parses ({} parses remain).".format(
            self.name(), len(bad_parses), len(good_parses)))    
        return good_parses, bad_parses

import sys
import re
from collections import defaultdict
from itertools import chain, product
from time import time
from multiprocessing import Process, Queue, JoinableQueue
from Queue import Empty

# from entity_features import compile_entity_feature_generator
# from snorkel import entity_internal
# from tree_structs import corenlp_to_xmltree, XMLTree

class Candidate(object):
    """A candidate object, **uniquely identified by its id**"""
    def __init__(self, id):
        self.id = id

    def __eq__(self, other):
        try:
            return self.id == other.id
        except:
            raise NotImplementedError()

    def __hash__(self):
        return hash(self.id)


class CandidateSpace(object):
    """
    Defines the **space** of candidate objects
    Calling _apply(x)_ given an object _x_ returns a generator over candidates in _x_.
    """
    def __init__(self):
        pass

    def apply(self, x):
        raise NotImplementedError()


class CandidateExtractorProcess(Process):
    def __init__(self, candidate_space, matcher, contexts_in, candidates_out):
        Process.__init__(self)
        self.candidate_space = candidate_space
        self.matcher         = matcher
        self.contexts_in     = contexts_in
        self.candidates_out  = candidates_out

    def run(self):
        while True:
            try:
                context = self.contexts_in.get(False)
                for candidate in self.matcher.apply(self.candidate_space.apply(context)):
                    self.candidates_out.put(candidate, False)
                self.contexts_in.task_done()
            except Empty:
                break


QUEUE_COLLECT_TIMEOUT = 5

class Candidates(object):
    """
    A generic class to hold and index a set of Candidates
    Takes in a CandidateSpace operator over some context type (e.g. Ngrams, applied over Sentence objects),
    a Matcher over that candidate space, and a set of context objects (e.g. Sentences)
    """
    def __init__(self, candidate_space, matcher, contexts, parallelism=False, join_key='context_id'):
        self.join_key = join_key
        self.ps = []
        self.feats = None
        self.feat_index = {}
        self.contexts = contexts

        # Extract & index candidates
        print "Extracting candidates..."
        if parallelism in [1, False]:
            candidates = self._extract(candidate_space, matcher, contexts)
        else:
            candidates = self._extract_multiprocess(candidate_space, matcher, contexts, parallelism=parallelism)
        self._index(candidates)

    def num_candidates(self):
        return len(self._candidates_by_id)

    def _extract(self, candidate_space, matcher, contexts):
        return chain.from_iterable(matcher.apply(candidate_space.apply(c)) for c in contexts)

    def _extract_multiprocess(self, candidate_space, matcher, contexts, parallelism=2):
        contexts_in    = JoinableQueue()
        candidates_out = Queue()

        # Fill the in-queue with contexts
        for context in contexts:
            contexts_in.put(context)

        # Start worker Processes
        for i in range(parallelism):
            p  = CandidateExtractorProcess(candidate_space, matcher, contexts_in, candidates_out)
            p.start()
            self.ps.append(p)

        # Join on JoinableQueue of contexts
        contexts_in.join()

        # Collect candidates out
        candidates = []
        while True:
            try:
                candidates.append(candidates_out.get(True, QUEUE_COLLECT_TIMEOUT))
            except Empty:
                break
        return candidates

    def _index(self, candidates):
        self._candidates_by_id         = {}
        self._candidates_by_context_id = defaultdict(list)
        for c in candidates:
            self._candidates_by_id[c.id] = c
            self._candidates_by_context_id[c.__dict__[self.join_key]].append(c)

    def __iter__(self):
        """Default iterator is over Candidates"""
        return self._candidates_by_id.itervalues()

    def get_candidates(self):
        return self._candidates_by_id.values()

    def get_candidate(self, id):
        """Retrieve a candidate by candidate id"""
        return self._candidates_by_id[id]

    def get_candidates_in(self, context_id):
        """Return the candidates in a specific context (e.g. Sentence)"""
        return self._candidates_by_context_id[context_id]

    def gold_stats(self, gold_set):
        """Return precision and recall relative to a "gold" set of candidates of the same type"""
        gold = gold_set if isinstance(gold_set, set) else set(gold_set)
        cs   = self.get_candidates()
        nc   = len(cs)
        ng   = len(gold)
        both = len(gold.intersection(cs))
        print "# of gold annotations\t= %s" % ng
        print "# of candidates\t\t= %s" % nc
        print "Candidate recall\t= %0.3f" % (both / float(ng),)
        print "Candidate precision\t= %0.3f" % (both / float(nc),)


# Basic sentence attributes
WORDS        = 'words'
CHAR_OFFSETS = 'char_offsets'
TEXT         = 'text'

class Ngram(Candidate):
    """A span of _n_ tokens, identified by sentence id and character-index start, end (inclusive)"""
    def __init__(self, char_start, char_end, sent, metadata={}):

        # Inherit full sentence object (tranformed to dict) and check for necessary attribs
        self.sentence = sent if isinstance(sent, dict) else sent._asdict()
        self.sent_id  = self.sentence['id']
        REQ_ATTRIBS = ['id', WORDS]
        if not all([self.sentence.has_key(a) for a in REQ_ATTRIBS]):
            raise ValueError("Sentence object must have attributes %s to form Ngram object" % ", ".join(REQ_ATTRIBS))

        # Set basic object attributes
        self.id          = "%s:%s-%s" % (self.sent_id, char_start, char_end)
        self.char_start  = char_start
        self.char_end    = char_end
        self.char_len    = char_end - char_start + 1
        self.word_start  = self.char_to_word_index(char_start)
        self.word_end    = self.char_to_word_index(char_end)
        self.n           = self.word_end - self.word_start + 1

        # NOTE: We assume that the char_offsets are **relative to the document start**
        self.sent_offset     = self.sentence[CHAR_OFFSETS][0]
        self.sent_char_start = self.char_start - self.sent_offset
        self.sent_char_end   = self.char_end - self.sent_offset

        # A dictionary to hold task-specific metadata e.g. canonical id, category, etc.
        self.metadata = metadata

        # To enable generic methods
        self.context_id = self.sent_id

    def __getitem__(self, key):
        """
        Slice operation returns a new candidate sliced according to **char index**
        Note that the slicing is w.r.t. the candidate range (not the abs. sentence char indexing)
        """
        if isinstance(key, slice):
            char_start = self.char_start if key.start is None else self.char_start + key.start
            if key.stop is None:
                char_end = self.char_end
            elif key.stop >= 0:
                char_end = self.char_start + key.stop - 1
            else:
                char_end = self.char_end + key.stop
            return Ngram(char_start, char_end, self.sentence)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return '<Ngram("%s", id=%s, chars=[%s,%s], words=[%s,%s])' \
            % (self.get_attrib_span(WORDS), self.id, self.char_start, self.char_end, self.word_start, self.word_end)

    def char_to_word_index(self, ci):
        """Given a character-level index (offset), return the index of the **word this char is in**"""
        for i,co in enumerate(self.sentence[CHAR_OFFSETS]):
            if ci == co:
                return i
            elif ci < co:
                return i-1
        return i

    def word_to_char_index(self, wi):
        """Given a word-level index, return the character-level index (offset) of the word's start"""
        return self.sentence[CHAR_OFFSETS][wi]

    def get_attrib_tokens(self, a):
        """Get the tokens of sentence attribute _a_ over the range defined by word_offset, n"""
        return self.sentence[a][self.word_start:self.word_end+1]

    def get_attrib_span(self, a, sep=" "):
        """Get the span of sentence attribute _a_ over the range defined by word_offset, n"""
        # NOTE: Special behavior for words currently (due to correspondence with char_offsets)
        if a == WORDS:
            return self.sentence[TEXT][self.sent_char_start:self.sent_char_end+1]
        else:
            return sep.join(self.get_attrib_tokens(a))

    def get_span(self, sep=" "):
        return self.get_attrib_span(WORDS)


    # def _apply(self, sent):
    #     xt = corenlp_to_xmltree(sent)
    #     for e_idxs, e_label in self.e.apply(sent):
    #         yield entity_internal(e_idxs, e_label, sent, xt)


class Ngrams(CandidateSpace):
    """
    Defines the space of candidates as all n-grams (n <= n_max) in a sentence _x_,
    indexing by **character offset**.
    """

    def __init__(self, n_max=5, split_tokens=['-', '/']):
        self.n_max        = n_max
        self.split_rgx    = r'('+r'|'.join(split_tokens)+r')' if split_tokens and len(split_tokens) > 0 else None

    def apply(self, x):
        s = x if isinstance(x, dict) else x._asdict()
        try:
            cos   = s[CHAR_OFFSETS]
            words = s[WORDS]
            text  = s[TEXT]
        except:
            raise ValueError("Input object must have attributes: " + ' '.join([CHAR_OFFSET, WORDS, TEXT]))


        # Loop over all n-grams in **reverse** order (to facilitate longest-match semantics)
        L = len(cos)
        for l in range(1, self.n_max+1)[::-1]:
            for i in range(L-l+1):
                cl         = cos[i+l-1] - cos[i] + len(words[i+l-1])
                char_start = cos[i]
                char_end   = cos[i] + cl - 1
                yield Ngram(char_start=char_start, char_end=char_end, sent=s)

                # Check for split
                # NOTE: For simplicity, we only split single tokens right now!
                if l == 1 and self.split_rgx is not None:
                    m = re.search(self.split_rgx, text[char_start:char_end+1])
                    if m is not None and l < self.n_max:
                        yield Ngram(char_start=char_start, char_end=char_start + m.start(1) - 1, sent=s)
                        yield Ngram(char_start=char_start + m.end(1), char_end=char_end, sent=s)


"""-------------------------HERE BE BRADEN'S KINGDOM-------------------------"""
# Basic table attributes
CELLS        = 'cells'

class CellNgram(Ngram):
    def __init__(self, cell, ngram):
        super(CellNgram, self).__init__(ngram.char_start, ngram.char_end, ngram.sentence)
        self.context_id = cell.context_id
        self.table_id = cell.table_id
        self.cell_id = cell.cell_id
        self.row_num = cell.row_num
        self.col_num = cell.col_num
        self.html_tag = cell.html_tag
        self.html_attrs = cell.html_attrs
        self.html_anc_tags = cell.html_anc_tags
        self.html_anc_attrs = cell.html_anc_attrs

    def __repr__(self):
        return '<CellNgram("%s", id=%s, chars=[%s,%s], (row,col)=(%s,%s), tag=%s)' \
            % (self.get_attrib_span(WORDS), self.id, self.char_start, self.char_end, self.row_num, self.col_num, self.html_tag)


class CellNgrams(Ngrams):
    """
    Defines the space of candidates as all n-grams (n <= n_max) in a cell within a table _x_
    "Calling _apply(x)_ given an object _x_ returns a generator over candidates in _x_."
    """
    def apply(self, x):
        table = x if isinstance(x, dict) else x._asdict()
        try:
            cells = table[CELLS]
        except:
            raise ValueError("Input object must have %s attribute" % CELLS)

        for cell in cells:
            for ngram in super(CellNgrams, self).apply(cell):
                yield CellNgram(cell, ngram)


# class EntityExtractor(object):
#     def __init__(self, candidate_space, matcher):
#         self.candidate_space = candidate_space
#         self.matcher = matcher

#     def apply(self, context):
#         # if 'table_id' in context._fields:
#         #     for cell in context.cells:
#         #         for e in self.matcher.apply(self.candidate_space.apply(cell)):
#         #             yield e
#         # else:
#         for e in self.matcher.apply(self.candidate_space.apply(context)):
#             yield e

# class RelationExtractor(object):
#     """
#     A generator for relation mentions. NOTE: currently limited to two entities/relation
#     """
#     def __init__(self, entity_extractors):
#         self.arity = len(entity_extractors) if isinstance(entity_extractors, list) else 1
#         self.extractors = entity_extractors

#     def apply(self, context):
#         """
#         Yield a relation for each cross-product (nested for loop) tuple of entities extracted
#             from the given context
#         """
#         for e1 in self.extractors[0].apply(context):
#             for e2 in self.extractors[1].apply(context):
#                 yield Relation(e1,e2)


# class Relation(Candidate):
#     def __init__(self, e1, e2):
#         self.id = "%s:%s" % (e1.id, e2.id)
#         self.context_id = e1.sent_id
#         self.e1 = e1
#         self.e2 = e2

#     def __repr__(self):
#         return 'Relation<Ngram("%s", id=%s),Ngram("%s", id=%s)>' \
#             % (self.e1.get_attrib_span(WORDS), self.e1.id, self.e2.get_attrib_span(WORDS), self.e2.id)

    # def _get_features(self):
    #     entity1_features
    #     entity2_features
    #     Cell_match = True
    #     Row_diff_low = False
    #     Row_diff_0 = True
    #     Row_diff_high = False
    #     Col_diff_low = False
    #     Col_diff_0 = True
    #     Col_diff_high = False
    #     [html_tag]_between = True    (e.g., hr, br)
    #     [Ngram]_between = True  (e.g., "Voltage")


# class Candidates(object):
#     """
#     A generic class to hold and index a set of Candidates
#     Takes in a CandidateSpace operator over some context type (e.g. Ngrams, applied over Sentence objects),
#     a Matcher over that candidate space, and a set of context objects (e.g. Sentences)
#     """
#     def __init__(self, extractor, contexts, parallelism=False, join_key='context_id'):
#         self.join_key = join_key
#         self.ps = []

#         # Extract & index candidates
#         print "Extracting candidates..."
#         if parallelism in [1, False]:
#             candidates = self._extract(extractor, contexts)
#         else:
#             candidates = self._extract_multiprocess(extractor, contexts)
#         self._index(candidates)

#     def _extract(self, extractor, contexts):
#         return chain.from_iterable(extractor.apply(c) for c in contexts)

#     def _extract_multiprocess(self, extractor, contexts, parallelism=2):
#         raise NotImplementedError

#     # NOTE: For tables, _get_features must have access to auxiliary data structures
#     def _get_features(self):
#         raise NotImplementedError

#     def _index(self, candidates):
#         self._candidates_by_id         = {}
#         self._candidates_by_context_id = defaultdict(list)
#         for c in candidates:
#             self._candidates_by_id[c.id] = c
#             self._candidates_by_context_id[c.__dict__[self.join_key]].append(c)

#     def __iter__(self):
#         """Default iterator is over Candidates"""
#         return self._candidates_by_id.itervalues()

#     def get_candidates(self):
#         return self._candidates_by_id.values()

#     def get_candidate(self, id):
#         """Retrieve a candidate by candidate id"""
#         return self._candidates_by_id[id]

#     def get_candidates_in(self, context_id):
#         """Return the candidates in a specific context (e.g. Sentence)"""
#         return self._candidates_by_context_id[context_id]

#     def gold_stats(self, gold_set):
#         """Return precision and recall relative to a "gold" set of candidates of the same type"""
#         gold = gold_set if isinstance(gold_set, set) else set(gold_set)
#         cs   = self.get_candidates()
#         nc   = len(cs)
#         ng   = len(gold)
#         both = len(gold.intersection(cs))
#         print "# of gold annotations\t= %s" % ng
#         print "# of candidates\t\t= %s" % nc
#         print "Candidate recall\t= %0.3f" % (both / float(ng),)
#         print "Candidate precision\t= %0.3f" % (both / float(nc),)

# class Entities(Candidates):
#     def __init__(self, entity_extractor, corpus, parallelism=False):
#         self.corpus = corpus
#         self.extractor = entity_extractor

#     def _extract(self, contexts):
#         return chain.from_iterable(self.extractor.apply(c) for c in contexts)

# class Relations(Candidates):
#     """
#     A generic class to hold and index a set of (candidate) Relations
#     Takes in a RelationExtractor and a corpus of context objects (e.g. Tables)
#     """
#     def __init__(self, relation_extractor, corpus, parallelism=False):
#         self.corpus = corpus
#         self.extractor = relation_extractor

#         # Extract & index candidates
#         print "Extracting candidates..."
#         if parallelism in [1, False]:
#             candidates = self._extract()
#         else:
#             candidates = self._extract_multiprocess()
#         self._index(candidates)

#     def _extract(self):
#         return chain.from_iterable(self.relation_extractor.apply(c) for c in self.corpus.get_contexts())

#     def _extract_multiprocess(self, relation_extractor, corpus, parallelism=2):
#         raise NotImplementedError

#     def _get_features(self):
#         raise NotImplementedError

#     def _index(self, candidates):
#         self._candidates_by_id         = {}
#         self._candidates_by_context_id = defaultdict(list)
#         for c in candidates:
#             self._candidates_by_id[c.id] = c
#             self._candidates_by_context_id[c.__dict__[self.join_key]].append(c)

#     def __iter__(self):
#         """Default iterator is over Candidates"""
#         return self._candidates_by_id.itervalues()

#     def get_candidates(self):
#         return self._candidates_by_id.values()

#     def get_candidate(self, id):
#         """Retrieve a candidate by candidate id"""
#         return self._candidates_by_id[id]

#     def get_candidates_in(self, context_id):
#         """Return the candidates in a specific context (e.g. Sentence)"""
#         return self._candidates_by_context_id[context_id]

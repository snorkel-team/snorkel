from .models import CandidateSet, TemporarySpan, Span, SpanPair
from itertools import chain
from multiprocessing import Process, Queue, JoinableQueue
from Queue import Empty
from copy import deepcopy
import re

QUEUE_COLLECT_TIMEOUT = 5


def gold_stats(candidates, gold):
        """Return precision and recall relative to a "gold" CandidateSet"""
        # TODO: Make this efficient via SQL
        nc   = len(candidates)
        ng   = len(gold)
        both = len(gold.intersection(candidates.candidates))
        print "# of gold annotations\t= %s" % ng
        print "# of candidates\t\t= %s" % nc
        print "Candidate recall\t= %0.3f" % (both / float(ng),)
        print "Candidate precision\t= %0.3f" % (both / float(nc),)


class CandidateSpace(object):
    """
    Defines the **space** of candidate objects
    Calling _apply(x)_ given an object _x_ returns a generator over candidates in _x_.
    """
    def __init__(self):
        pass

    def apply(self, x):
        raise NotImplementedError()


class CandidateExtractor(object):
    # TODO: Revise docstring!
    """
    A generic class to create a Candidates object, which is a set of Candidate objects.

    Takes in a CandidateSpace operator over some context type (e.g. Ngrams, applied over Sentence objects),
    a Matcher over that candidate space, and a set of context objects (e.g. Sentences)
    """
    def __init__(self, cspaces, matchers, join_fn=None, no_nesting=True):
        self.candidate_spaces = cspaces if type(cspaces) in [list, tuple] else [cspaces]
        self.matchers         = matchers if type(matchers) in [list, tuple] else [matchers]
        self.join_fn          = join_fn
        self.no_nesting       = no_nesting

        # Check that arity is same
        if len(self.candidate_spaces) != len(self.matchers):
            raise ValueError("Mismatched arity of candidate space and matcher.")
        else:
            self.arity = len(self.candidate_spaces)

        # Check for whether it is a self-relation
        self.self_relation = False
        if self.arity == 2:
            if self.candidate_spaces[0] == self.candidate_spaces[1] and self.matchers[0] == self.matchers[1]:
                self.self_relation = True

        # Make sure the candidate spaces are different so generators aren't expended!
        self.candidate_spaces = map(deepcopy, self.candidate_spaces)

        # Track processes for multicore execution
        self.ps = []

    def extract(self, contexts, name, session=None, parallelism=False):

        # Create a candidate set
        c = CandidateSet(name=name)
        if session is not None:
            session.add(c)

        # If arity > 1, create a unary candidate set as well
        unary_set = CandidateSet(name=str(name + '-unary')) if self.arity > 1 else None

        # Run extraction
        if parallelism in [1, False]:
            for context in contexts:
                for candidate in self._extract_from_context(context, unary_set=unary_set):
                    c.candidates.append(candidate)
        else:
            for candidate in self._extract_multiprocess(contexts, parallelism, name):
                c.candidates.append(candidate)

        # Commit the session and return the candidate set
        if session is not None:
            session.commit()
        return c

    def _extract_from_context(self, context, unary_set=None):

        # Unary candidates
        if self.arity == 1:
            for tc in self.matchers[0].apply(self.candidate_spaces[0].apply(context)):
                yield tc.promote()

        # Binary candidates
        elif self.arity == 2:

            # Materialize once if self-relation; we materialize assuming that we have small contexts s.t.
            # computation expense > memory expense
            if self.self_relation:
                tcs1 = list(self.matchers[0].apply(self.candidate_spaces[0].apply(context)))
                tcs2 = tcs1
            else:
                tcs1 = list(self.matchers[0].apply(self.candidate_spaces[0].apply(context)))
                tcs2 = list(self.matchers[1].apply(self.candidate_spaces[1].apply(context)))

            # Do the local join, materializing all pairs of matched unary candidates
            promoted_candidates = {}
            for tc1 in tcs1:
                for tc2 in tcs2:

                    # Check for self-joins and "nested" joins (joins from span to its subspan)
                    if tc1 == tc2 or (self.no_nesting and (tc1 in tc2 or tc2 in tc1)):
                        continue

                    # AND-composition of implicit context.id join with optional join_fn condition
                    if (self.join_fn is None or self.join_fn(tc1, tc2)):
                        if tc1 not in promoted_candidates:
                            promoted_candidates[tc1] = tc1.promote()
                            promoted_candidates[tc1].set = span_set

                        if tc2 not in promoted_candidates:
                            promoted_candidates[tc2] = tc2.promote()
                            promoted_candidates[tc2].set = span_set

                        c1 = promoted_candidates[tc1]
                        c2 = promoted_candidates[tc2]
                        
                        # TODO: Un-hardcode this!
                        if isinstance(c1, Span) and isinstance(c2, Span):
                            yield SpanPair(span0=c1, span1=c2)
                        else:
                            raise NotImplementedError("Only Spans -> SpanPair mappings are handled currently.")

        # Higher-arity candidates
        else:
            raise NotImplementedError()
            
    def _extract_multiprocess(self, contexts, parallelism, name=None):
        contexts_in    = JoinableQueue()
        candidates_out = Queue()

        # Fill the in-queue with contexts
        for context in contexts:
            contexts_in.put(context)

        # Start worker Processes
        for i in range(parallelism):
            p = CandidateExtractorProcess(self._extract_from_context, contexts_in, candidates_out, name=name)
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


class CandidateExtractorProcess(Process):
    def __init__(self, extractor, contexts_in, candidates_out, name=None):
        Process.__init__(self)
        self.extractor      = extractor
        self.contexts_in    = contexts_in
        self.candidates_out = candidates_out
        self.name           = name

    def run(self):
        while True:
            try:
                context = self.contexts_in.get(False)
                for candidate in self.extractor(context, self.name):
                    self.candidates_out.put(candidate, False)
                self.contexts_in.task_done()
            except Empty:
                break


class Ngrams(CandidateSpace):
    """
    Defines the space of candidates as all n-grams (n <= n_max) in a Sentence _x_,
    indexing by **character offset**.
    """
    def __init__(self, n_max=5, split_tokens=['-', '/']):
        CandidateSpace.__init__(self)
        self.n_max = n_max
        self.split_rgx    = r'('+r'|'.join(split_tokens)+r')' if split_tokens and len(split_tokens) > 0 else None
    
    def apply(self, context):
        # Loop over all n-grams in **reverse** order (to facilitate longest-match semantics)
        L = len(context.char_offsets)
        for l in range(1, self.n_max+1)[::-1]:
            for i in range(L-l+1):
                # NOTE that we derive char_len without using sep
                char_start = context.char_offsets[i]
                cl = context.char_offsets[i+l-1] - context.char_offsets[i] + len(context.words[i+l-1])
                char_end = context.char_offsets[i] + cl - 1
                yield TemporarySpan(char_start=char_start, char_end=char_end, context=context)

                # Check for split
                # NOTE: For simplicity, we only split single tokens right now!
                if l == 1 and self.split_rgx is not None:
                    m = re.search(self.split_rgx, context.text[char_start-context.char_offsets[0]:char_end-context.char_offsets[0]+1])
                    if m is not None and l < self.n_max:
                        yield TemporarySpan(char_start=char_start, char_end=char_start + m.start(1) - 1, context=context)
                        yield TemporarySpan(char_start=char_start + m.end(1), char_end=char_end, context=context)


class NgramsOld(CandidateSpace):
    """
    Defines the space of candidates as all n-grams (n <= n_max) in a sentence _x_,
    indexing by **character offset**.
    """
    def __init__(self, n_max=5, split_tokens=['-', '/']):
        CandidateSpace.__init__(self)
        self.n_max        = n_max
        self.split_rgx    = r'('+r'|'.join(split_tokens)+r')' if split_tokens and len(split_tokens) > 0 else None
    
    def apply(self, x):
        s = get_as_dict(x)
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
                    m = re.search(self.split_rgx, text[char_start-cos[0]:char_end-cos[0]+1])
                    if m is not None and l < self.n_max:
                        yield Ngram(char_start=char_start, char_end=char_start + m.start(1) - 1, sent=s)
                        yield Ngram(char_start=char_start + m.end(1), char_end=char_end, sent=s)




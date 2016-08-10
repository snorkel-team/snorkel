from .models import CandidateSet, TemporarySpan, Span, SpanPair
from itertools import chain
from multiprocessing import Process, Queue, JoinableQueue
from Queue import Empty

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


class CandidateExtractor(object):
    # TODO: Revise docstring!
    """
    A generic class to create a Candidates object, which is a set of Candidate objects.

    Takes in a CandidateSpace operator over some context type (e.g. Ngrams, applied over Sentence objects),
    a Matcher over that candidate space, and a set of context objects (e.g. Sentences)
    """
    def __init__(self, cspaces, matchers, parallelism=False, join_fn=None, no_nesting=True):
        self.candidate_spaces = cspaces if type(cspaces) in [list, tuple] else [cspaces]
        self.matchers         = matchers if type(matchers) in [list, tuple] else [matchers]
        self.parallelism      = parallelism
        self.join_fn          = join_fn
        self.no_nesting       = no_nesting

        # Check that arity is same
        if len(self.candidate_spaces) != len(self.matchers):
            raise ValueError("Mismatched arity of candidate space and matcher.")
        else:
            self.arity = len(self.candidate_spaces)

        # Track processes for multicore execution
        self.ps = []

    def extract(self, contexts, name=None):
        c = CandidateSet()
        if self.parallelism in [1, False]:
            for candidate in self._extract(contexts):
                c.candidates.append(candidate)
        else:
            for candidate in self._extract_multiprocess(contexts):
                c.candidates.append(candidate)
        if name is not None:
            c.name = name
        return c

    def _extract(self, contexts):

        # Unary candidates
        if self.arity == 1:
            for tc in chain.from_iterable(self.matchers[0].apply(self.candidate_spaces[0].apply(context)) \
                for context in contexts):
                yield tc.promote()

        # Binary candidates
        elif self.arity == 2:
            
            # NOTE: We are implicitly joining on context1.id == context2.id by iterating over contexts!
            for context in contexts:

                # Self-relations- materialize once to avoid repeated computation
                if self.candidate_spaces[0] == self.candidate_spaces[1] and self.matchers[0] == self.matchers[1]:
                    tcs1 = list(self.matchers[0].apply(self.candidate_spaces[0].apply(context)))
                    tcs2 = tcs1
                else:
                    tcs1 = self.matchers[0].apply(self.candidate_spaces[0].apply(context))
                    tcs2 = self.matchers[1].apply(self.candidate_spaces[1].apply(context))

                # Do the local join, materializing all pairs of matched unary candidates
                for tc1 in tcs1:
                    for tc2 in tcs2:

                        # Check for self-joins and "nested" joins (joins from span to its subspan)
                        if tc1 == tc2 or (self.no_nesting and (tc1 in tc2 or tc2 in tc1)):
                            continue

                        # AND-composition of implicit context.id join with optional join_fn condition
                        if (self.join_fn is None or self.join_fn(tc1, tc2)):
                            c1 = tc1.promote()
                            c2 = tc2.promote()
                            
                            # TODO: Un-hardcode this!
                            if isinstance(c1, Span) and isinstance(c2, Span):
                                yield SpanPair(span0=c1, span1=c2)
                            else:
                                raise NotImplementedError("Only Spans -> SpanPair mappings are handled currently.")

        # Higher-arity candidates
        else:
            raise NotImplementedError()
            
    def _extract_multiprocess(self, contexts):
        # TODO: Implement this!!!
        raise NotImplementedError()
        contexts_in    = JoinableQueue()
        candidates_out = Queue()

        # Fill the in-queue with contexts
        for context in contexts:
            contexts_in.put(context)

        # Start worker Processes
        for i in range(self.parallelism):
            p = CandidateExtractorProcess(self.candidate_space, self.matcher, contexts_in, candidates_out)
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

class Ngrams(CandidateSpace):
    """
    Defines the space of candidates as all n-grams (n <= n_max) in a Sentence _x_,
    indexing by **character offset**.
    """
    def __init__(self, n_max=5):
        CandidateSpace.__init__(self)
        self.n_max = n_max
    
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

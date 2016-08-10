from .models import CandidateSet, TemporarySpan
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
    """
    A generic class to create a Candidates object, which is a set of Candidate objects.

    Takes in a CandidateSpace operator over some context type (e.g. Ngrams, applied over Sentence objects),
    a Matcher over that candidate space, and a set of context objects (e.g. Sentences)
    """
    def __init__(self, candidate_space, matcher, parallelism=False, join_key='context_id'):
        self.candidate_space = candidate_space
        self.matcher = matcher
        self.parallelism = parallelism
        self.join_key = join_key

        self.ps = []

    def extract(self, contexts, name=None):
        c = CandidateSet()

        if self.parallelism in [1, False]:
            for candidate in self._extract(contexts):
                c.candidates.append(candidate.promote())
        else:
            for candidate in self._extract_multiprocess(contexts):
                c.candidates.append(candidate.promote())

        if name is not None:
            c.name = name

        return c

    def _extract(self, contexts):
        return chain.from_iterable(self.matcher.apply(self.candidate_space.apply(c)) for c in contexts)

    def _extract_multiprocess(self, contexts):
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

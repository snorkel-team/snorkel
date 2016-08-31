from . import SnorkelSession
from .models import CandidateSet, TemporarySpan, Span
from itertools import chain
from multiprocessing import Process, Queue, JoinableQueue
from Queue import Empty
from copy import deepcopy
import re

QUEUE_COLLECT_TIMEOUT = 5

PTB = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}', '-LCB-': '{',
         '-RSB-': ']', '-LSB-': '['}

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
    def __init__(self, candidate_class, cspaces, matchers, join_fn=None, self_relations=False, nested_relations=False, symmetric_relations=True):
        self.candidate_class     = candidate_class
        self.candidate_spaces    = cspaces if type(cspaces) in [list, tuple] else [cspaces]
        self.matchers            = matchers if type(matchers) in [list, tuple] else [matchers]
        self.join_fn             = join_fn
        self.nested_relations    = nested_relations
        self.self_relations      = self_relations
        self.symmetric_relations = symmetric_relations

        # Check that arity is same
        if len(self.candidate_spaces) != len(self.matchers):
            raise ValueError("Mismatched arity of candidate space and matcher.")
        else:
            self.arity = len(self.candidate_spaces)

        # Make sure the candidate spaces are different so generators aren't expended!
        self.candidate_spaces = map(deepcopy, self.candidate_spaces)

        # Track processes for multicore execution
        self.ps = []

    def extract(self, contexts, name, session, parallelism=False):
        # Create a candidate set
        c = CandidateSet(name=name)
        session.add(c)
        session.commit()

        # Run extraction
        if parallelism in [1, False]:
            for context in contexts:
                for candidate in self._extract_from_context(context):
                    session.add(candidate)
                    c.candidates.append(candidate)

            # Commit the candidates and return the candidate set
            session.commit()
            return c
        else:
            self._extract_multiprocess(contexts, c, parallelism)
            return session.query(CandidateSet).filter(CandidateSet.name == name).one()

    def _extract_from_context(self, context, unary_set=None):
        temp_spans = [None] * self.arity
        for i in range(self.arity):
            temp_spans[i] = self._generate_temp_spans(context, self.candidate_spaces[i], self.matchers[i])


        # Unary candidates
        if self.arity == 1:
            for tc in self.matchers[0].apply(self.candidate_spaces[0].apply(context)):
                yield tc.promote()

        # Binary candidates
        elif self.arity == 2:

            # Materialize once if self-relation; we materialize assuming that we have small contexts s.t.
            # computation expense > memory expense
            if self.same_unary:
                tcs1 = list(self.matchers[0].apply(self.candidate_spaces[0].apply(context)))
                tcs2 = tcs1
            else:
                tcs1 = list(self.matchers[0].apply(self.candidate_spaces[0].apply(context)))
                tcs2 = list(self.matchers[1].apply(self.candidate_spaces[1].apply(context)))

            # Do the local join, materializing all pairs of matched unary candidates
            promoted_candidates = {}
            for i,tc1 in enumerate(tcs1):
                for j,tc2 in enumerate(tcs2):

                    # Optionally exclude symmetric relations; equivalent to extracting *undirected* relations
                    if self.same_unary and not self.symmetric_relations and j < i:
                        continue

                    # Check for self-joins and "nested" joins (joins from span to its subspan)
                    if not self.self_relations and tc1 == tc2:
                        continue
                    if not self.nested_relations and (tc1 in tc2 or tc2 in tc1):
                        continue

                    # AND-composition of implicit context.id join with optional join_fn condition
                    if (self.join_fn is None or self.join_fn(tc1, tc2)):
                        if tc1 not in promoted_candidates:
                            promoted_candidates[tc1] = tc1.promote()
                            promoted_candidates[tc1].set = unary_set

                        if tc2 not in promoted_candidates:
                            promoted_candidates[tc2] = tc2.promote()
                            promoted_candidates[tc2].set = unary_set

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
            
    def _extract_multiprocess(self, contexts, candidate_set, parallelism):
        contexts_in    = JoinableQueue()
        candidates_out = Queue()

        # Fill the in-queue with contexts
        for context in contexts:
            contexts_in.put(context)

        # Start worker Processes
        for i in range(parallelism):
            session = SnorkelSession()
            c = session.merge(candidate_set)
            p = CandidateExtractorProcess(self._extract_from_context, session, contexts_in, candidates_out, c)
            self.ps.append(p)

        for p in self.ps:
            p.start()
        
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

    def _generate_temp_spans(self, context, space, matcher, output_set):
        """
        Generates TemporarySpans for a context, using the provided space and matcher

        :param context: the context for which temporary spans will be generated
        :param space: the space of TemporarySpans to consider
        :param matcher: the matcher that the TemporarySpans must pass to be returned
        :param output_set: set container in which to store output. This method will clear it before using
        :return: set of unique TemporarySpans, container passed in as output_set
        """
        pass

    def _persist_spans(self, temp_span_list, session, output_list):
        """
        Given a list of sets of TemporarySpans, produces a list of sets of Spans persisted
        in the database, such that each TemporarySpan has a corresponding Span

        :param temp_span_list: list of sets of TemporarySpans to persist
        :param session: the Session to use to access the database
        :param output_list: list of sets in which to store output. This method will clear each before using
        :return: list of sets of persisted Spans, container passed in as output_list
        """
        pass

class CandidateExtractorProcess(Process):
    def __init__(self, extractor, session, contexts_in, candidates_out, candidate_set, unary_set):
        Process.__init__(self)
        self.extractor      = extractor
        self.session        = session
        self.contexts_in    = contexts_in
        self.candidates_out = candidates_out
        self.candidate_set  = candidate_set
        self.unary_set      = unary_set

    def run(self):
        c = self.candidate_set
        u = self.unary_set if self.unary_set is not None else None

        unique_candidates = set()
        while True:
            try:
                context = self.session.merge(self.contexts_in.get(False))
                for candidate in self.extractor(context, u):
                    unique_candidates.add(candidate)

                for candidate in unique_candidates:
                    c.candidates.append(candidate)

                unique_candidates.clear()
                self.contexts_in.task_done()
            except Empty:
                break

        self.session.commit()
        self.session.close()


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
                
                w = context.words[i+l-1] if context.words[i+l-1] not in PTB else PTB[context.words[i+l-1]]
                
                
                cl = context.char_offsets[i+l-1] - context.char_offsets[i] + len(w)
                char_end = context.char_offsets[i] + cl - 1
                yield TemporarySpan(char_start=char_start, char_end=char_end, context=context)

                # Check for split
                # NOTE: For simplicity, we only split single tokens right now!
                if l == 1 and self.split_rgx is not None:
                    m = re.search(self.split_rgx, context.text[char_start-context.char_offsets[0]:char_end-context.char_offsets[0]+1])
                    if m is not None and l < self.n_max + 1:
                        yield TemporarySpan(char_start=char_start, char_end=char_start + m.start(1) - 1, context=context)
                        yield TemporarySpan(char_start=char_start + m.end(1), char_end=char_end, context=context)

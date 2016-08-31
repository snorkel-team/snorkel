from . import SnorkelSession
from .models import CandidateSet, TemporarySpan, Context
from itertools import product
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

        # Preallocates internal data structures
        self.child_context_sets = [None] * self.arity
        for i in range(self.arity):
            self.child_context_sets[i] = set()

        # Track processes for multicore execution
        self.ps = []

    def extract(self, contexts, name, session, parallelism=False):
        # Create a candidate set
        c = CandidateSet(name=name)
        session.add(c)

        # Run extraction
        if parallelism in [1, False]:
            for context in contexts:
                for candidate in self._extract_from_context(context, session):
                    c.candidates.append(candidate)

            # Commit the candidates and return the candidate set
            session.commit()
            return c
        else:
            session.commit()
            self._extract_multiprocess(contexts, c, parallelism)
            return session.query(CandidateSet).filter(CandidateSet.name == name).one()

    def _extract_from_context(self, context, session):
        # Applies matchers to generate child contexts
        for i in range(self.arity):
            self._generate_child_contexts(context, self.candidate_spaces[i], self.matchers[i], session, self.child_context_sets[i])

        # Generates and persists candidates
        filter_map = {}
        arg_map = {}
        arg_names = self.candidate_class.__argnames__
        for args in product(*[enumerate(child_contexts) for child_contexts in self.child_context_sets]):
            # Check for self-joins and "nested" joins (joins from span to its subspan)
            if self.arity == 2 and not self.self_relations and args[0][1] == args[1][1]:
                continue
            # TODO: Make this work for higher-order relations
            if self.arity == 2 and not self.nested_relations and (args[0][1] in args[1][1] or args[1][1] in args[0][1]):
                continue
            # Checks for symmetric relations
            if self.arity == 2 and not self.symmetric_relations and args[0][0] > args[1][0]:
                continue

            for i, arg_name in enumerate(arg_names):
                filter_map[arg_name] = args[i][1]
                # scrap: self.candidate_class.__name__ + '.' +
                arg_map[arg_name] = args[i][1]
            candidate = session.query(self.candidate_class).filter_by(**filter_map).first()
            if candidate is None:
                candidate = self.candidate_class(**arg_map)
                session.add(candidate)
            yield candidate
            
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

    def _generate_child_contexts(self, context, candidate_space, matcher, session, output_contexts):
        """
        Generates TemporaryContexts for a context, using the provided space and matcher

        :param context: the context for which temporary spans will be generated
        :param candidate_space: the space of TemporarySpans to consider
        :param matcher: the matcher that the TemporarySpans must pass to be returned
        :param session: the session
        :param output_contexts: set container in which to store output. This method will clear it before using
        """
        output_contexts.clear()

        # Generate TemporaryContexts that are children of the context using the candidate_space and filtered
        # by the Matcher
        for tc in matcher.apply(candidate_space.apply(context)):

            # Query the database to see if this context exists already
            c = session.query(Context).filter(Context.stable_id == tc.get_stable_id()).first()
            if c is None:
                c = tc.promote()
                session.add(c)
            output_contexts.add(c)


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


class CandidateSpace(object):
    """
    Defines the **space** of candidate objects
    Calling _apply(x)_ given an object _x_ returns a generator over candidates in _x_.
    """
    def __init__(self):
        pass

    def apply(self, x):
        raise NotImplementedError()


class Ngrams(CandidateSpace):
    """
    Defines the space of candidates as all n-grams (n <= n_max) in a Sentence _x_,
    indexing by **character offset**.
    """
    def __init__(self, n_max=5, split_tokens=['-', '/']):
        CandidateSpace.__init__(self)
        self.n_max     = n_max
        self.split_rgx = r'('+r'|'.join(split_tokens)+r')' if split_tokens and len(split_tokens) > 0 else None
    
    def apply(self, context):

        # These are the character offset--**relative to the sentence start**--for each _token_
        offsets = context.char_offsets

        # Loop over all n-grams in **reverse** order (to facilitate longest-match semantics)
        L    = len(offsets)
        seen = set()
        for l in range(1, self.n_max+1)[::-1]:
            for i in range(L-l+1):
                w     = context.words[i+l-1]
                start = offsets[i]
                end   = offsets[i+l-1] + len(w) - 1
                ts    = TemporarySpan(char_start=start, char_end=end, parent=context)
                if ts not in seen:
                    seen.add(ts)
                    yield ts

                # Check for split
                # NOTE: For simplicity, we only split single tokens right now!
                if l == 1 and self.split_rgx is not None:
                    m = re.search(self.split_rgx, context.text[start-offsets[0]:end-offsets[0]+1])
                    if m is not None and l < self.n_max + 1:
                        ts1 = TemporarySpan(char_start=start, char_end=start + m.start(1) - 1, parent=context)
                        if ts1 not in seen:
                            seen.add(ts1)
                            yield ts
                        ts2 = TemporarySpan(char_start=start + m.end(1), char_end=end, parent=context)
                        if ts2 not in seen:
                            seen.add(ts2)
                            yield ts2

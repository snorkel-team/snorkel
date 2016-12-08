from . import SnorkelSession
from .utils import ProgressBar
from .models import Candidate, TemporarySpan, Sentence, Span
from itertools import product
from multiprocessing import Process, Queue, JoinableQueue
from sqlalchemy.sql import select
from Queue import Empty
from copy import deepcopy
import re
from collections import defaultdict

QUEUE_COLLECT_TIMEOUT = 5


class CandidateExtractor(object):
    """
    An operator to extract Candidate objects from Context objects.

    :param candidate_class: The type of relation to extract, defined using
                            :func:`snorkel.models.candidate_subclass <snorkel.models.candidate.candidate_subclass>`
    :param cspaces: one or list of :class:`CandidateSpace` objects, one for each relation argument. Defines space of
                    Contexts to consider
    :param matchers: one or list of :class:`snorkel.matchers.Matcher` objects, one for each relation argument. Only tuples of
                     Contexts for which each element is accepted by the corresponding Matcher will be returned as Candidates
    :param self_relations: Boolean indicating whether to extract Candidates that relate the same context.
                           Only applies to binary relations. Default is False.
    :param nested_relations: Boolean indicating whether to extract Candidates that relate one Context with another
                             that contains it. Only applies to binary relations. Default is False.
    :param symmetric_relations: Boolean indicating whether to extract symmetric Candidates, i.e., rel(A,B) and rel(B,A),
                                where A and B are Contexts. Only applies to binary relations. Default is True.
    """
    def __init__(self, candidate_class, cspaces, matchers, self_relations=False, nested_relations=False, symmetric_relations=True):
        self.candidate_class     = candidate_class
        self.candidate_spaces    = cspaces if type(cspaces) in [list, tuple] else [cspaces]
        self.matchers            = matchers if type(matchers) in [list, tuple] else [matchers]
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

    def extract(self, session, contexts, split=0, return_all=False):
        pb = ProgressBar(len(contexts))
        for i, context in enumerate(contexts):
            pb.bar(i)
            self._extract_from_context(session, context, split)
        pb.close()
        session.commit()

        # Optionally return full set of candidates
        if return_all:
            return session.query(self.candidate_class).filter(self.candidate_class.split == split).all()

    def _extract_from_context(self, session, context, split=0, check_for_existing=True):
        # Generate TemporaryContexts that are children of the context using the candidate_space and filtered
        # by the Matcher
        for i in range(self.arity):
            self.child_context_sets[i].clear()
            for tc in self.matchers[i].apply(self.candidate_spaces[i].apply(context)):
                tc.load_id_or_insert(session)
                self.child_context_sets[i].add(tc)

        # Generates and persists candidates
        candidate_args = {'split' : split}
        for args in product(*[enumerate(child_contexts) for child_contexts in self.child_context_sets]):

            # TODO: Make this work for higher-order relations
            if self.arity == 2:
                ai, a = args[0]
                bi, b = args[1]

                # Check for self-joins, "nested" joins (joins from span to its subspan), and flipped duplicate
                # "symmetric" relations
                if not self.self_relations and a == b:
                    continue
                elif not self.nested_relations and (a in b or b in a):
                    continue
                elif not self.symmetric_relations and ai > bi:
                    continue

            # Assemble candidate arguments
            for i, arg_name in enumerate(self.candidate_class.__argnames__):
                candidate_args[arg_name + '_id'] = args[i][1].id

            # Checking for existence
            if check_for_existing:
                q = select([self.candidate_class.id])
                for key, value in candidate_args.items():
                    q = q.where(getattr(self.candidate_class, key) == value)
                candidate_id = session.execute(q).first()
                if candidate_id is not None:
                    continue

            # Add Candidate to session
            session.add(self.candidate_class(**candidate_args))
            

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


class PretaggedCandidateExtractor(object):
    """
    An extractor for Sentences with entities pre-tagged, and stored in the entity_types and entity_cids
    fields.
    """
    def __init__(self, candidate_class, entity_types, self_relations=False, nested_relations=False, symmetric_relations=True, entity_sep='~@~'):
        self.candidate_class     = candidate_class
        self.entity_types        = entity_types
        self.arity               = len(entity_types)
        self.self_relations      = self_relations
        self.nested_relations    = nested_relations
        self.symmetric_relations = symmetric_relations
        self.entity_sep          = entity_sep

    def extract(self, session, contexts, split=0, return_all=False):
        """Extract Candidates locally from each Context in a provided list of Contexts"""
        pb = ProgressBar(len(contexts))
        for i, context in enumerate(contexts):
            pb.bar(i)
            self._extract_from_context(session, context, split)
        pb.close()
        session.commit()

        # Optionally return full set of candidates
        if return_all:
            return session.query(Candidate).filter(Candidate.split == split).all()

    def _extract_from_context(self, session, context, split=0):
        """
        Extract Candidates from a Context, and add to CandidateSet c
        Do so 
        """
        # For now, just handle Sentences
        if not isinstance(context, Sentence):
            raise NotImplementedError("%s is currently only implemented for Sentence contexts." % self.__name__)

        # Do a first pass to collect all mentions by entity type / cid
        entity_idxs = dict((et, defaultdict(list)) for et in set(self.entity_types))
        L = len(context.words)
        for i in range(L):
            if context.entity_types[i] is not None:
                ets  = context.entity_types[i].split(self.entity_sep)
                cids = context.entity_cids[i].split(self.entity_sep)
                for et, cid in zip(ets, cids):
                    if et in entity_idxs:
                        entity_idxs[et][cid].append(i)

        # Form entity Spans
        entity_spans = defaultdict(list)
        entity_cids  = {}
        for et, cid_idxs in entity_idxs.iteritems():
            for cid, idxs in entity_idxs[et].iteritems():
                while len(idxs) > 0:
                    i          = idxs.pop(0)
                    char_start = context.char_offsets[i]
                    char_end   = char_start + len(context.words[i]) - 1
                    while len(idxs) > 0 and idxs[0] == i + 1:
                        i        = idxs.pop(0)
                        char_end = context.char_offsets[i] + len(context.words[i]) - 1

                    # Insert / load temporary span, also store map to entity CID
                    tc = TemporarySpan(char_start=char_start, char_end=char_end, parent=context)
                    tc.load_id_or_insert(session)
                    entity_cids[tc.id] = cid
                    entity_spans[et].append(tc)

        # Generates and persists candidates
        parent_insert_query = Candidate.__table__.insert()
        parent_insert_args = {'type': self.candidate_class.__mapper_args__['polymorphic_identity']}
        arg_names = self.candidate_class.__argnames__
        child_insert_query = self.candidate_class.__table__.insert()
        child_args = {'split' : split}
        for args in product(*[enumerate(entity_spans[et]) for et in self.entity_types]):

            # Check for self-joins and "nested" joins (joins from span to its subspan)
            if self.arity == 2 and not self.self_relations and args[0][1] == args[1][1]:
                continue

            # TODO: Make this work for higher-order relations
            if self.arity == 2 and not self.nested_relations and (args[0][1] in args[1][1] or args[1][1] in args[0][1]):
                continue

            # Checks for symmetric relations
            if self.arity == 2 and not self.symmetric_relations and args[0][0] > args[1][0]:
                continue

            # Set Candidate Spans
            for i, arg_name in enumerate(arg_names):
                child_args[arg_name + '_id'] = args[i][1].id

            # Set Candidate CIDS
            for i, arg_name in enumerate(arg_names):
                child_args[arg_name + '_cid'] = entity_cids[args[i][1].id]

            # See if candidate exists
            q = select([self.candidate_class.id])
            for key, value in child_args.items():
                q = q.where(getattr(self.candidate_class, key) == value)
            candidate_id = session.execute(q).first()

            # If candidate does not exist, persists it
            if candidate_id is None:
                candidate_id = session.execute(parent_insert_query, parent_insert_args).inserted_primary_key
                child_args['id'] = candidate_id[0]
                session.execute(child_insert_query, child_args)
                del child_args['id']

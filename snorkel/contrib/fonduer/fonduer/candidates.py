from ....candidates import CandidateSpace, Ngrams
from ....models.context import Document

from .models import TemporaryImage

from copy import deepcopy
from itertools import product
from sqlalchemy.sql import select

from ....models import Candidate
from ....udf import UDF, UDFRunner


class CandidateExtractor(UDFRunner):
    """An operator to extract Candidate objects from a Context.

    :param candidate_class: The type of relation to extract, defined using
                            :func:`snorkel.models.candidate_subclass <snorkel.models.candidate.candidate_subclass>`
    :param cspaces: one or list of :class:`CandidateSpace` objects, one for each relation argument. Defines space of
                    Contexts to consider
    :param matchers: one or list of :class:`snorkel.matchers.Matcher` objects, one for each relation argument. Only tuples of
                     Contexts for which each element is accepted by the corresponding Matcher will be returned as Candidates
    :param candidate_filter: an optional function for filtering out candidates which returns a Boolean expressing whether or not
                      the candidate should be instantiated.
    :param self_relations: Boolean indicating whether to extract Candidates that relate the same context.
                           Only applies to binary relations. Default is False.
    :param nested_relations: Boolean indicating whether to extract Candidates that relate one Context with another
                             that contains it. Only applies to binary relations. Default is False.
    :param symmetric_relations: Boolean indicating whether to extract symmetric Candidates, i.e., rel(A,B) and rel(B,A),
                                where A and B are Contexts. Only applies to binary relations. Default is True.
    """

    def __init__(self, candidate_class, cspaces, matchers, candidate_filter=None, self_relations=False, nested_relations=False, symmetric_relations=True):
        """Initialize the CandidateExtractor."""
        super(CandidateExtractor, self).__init__(CandidateExtractorUDF,
                                                 candidate_class=candidate_class,
                                                 cspaces=cspaces,
                                                 matchers=matchers,
                                                 candidate_filter=candidate_filter,
                                                 self_relations=self_relations,
                                                 nested_relations=nested_relations,
                                                 symmetric_relations=symmetric_relations)

    def apply(self, xs, split=0, **kwargs):
        """Call the CandidateExtractorUDF."""
        super(CandidateExtractor, self).apply(xs, split=split, **kwargs)

    def clear(self, session, split, **kwargs):
        """Delete Candidates from given split the database."""
        session.query(Candidate).filter(Candidate.split == split).delete()


class CandidateExtractorUDF(UDF):
    """UDF for performing candidate extraction."""

    def __init__(self, candidate_class, cspaces, matchers, candidate_filter, self_relations, nested_relations, symmetric_relations, **kwargs):
        """Initialize the CandidateExtractorUDF."""
        self.candidate_class     = candidate_class
        self.candidate_spaces    = cspaces if type(cspaces) in [list, tuple] else [cspaces]
        self.matchers            = matchers if type(matchers) in [list, tuple] else [matchers]
        self.candidate_filter    = candidate_filter
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

        super(CandidateExtractorUDF, self).__init__(**kwargs)

    def apply(self, context, clear, split, **kwargs):
        """Extract candidates from the given Context.

        Here, we define a context as a Phrase.
        :param context:
        :param clear:
        :param split: Which split to use.
        """
        # Generate TemporaryContexts that are children of the context using the candidate_space and filtered
        # by the Matcher
        for i in range(self.arity):
            self.child_context_sets[i].clear()
            for tc in self.matchers[i].apply(self.candidate_spaces[i].apply(self.session, context)):
                tc.load_id_or_insert(self.session)
                self.child_context_sets[i].add(tc)

        # Generates and persists candidates
        candidate_args = {'split': split}
        for args in product(*[enumerate(child_contexts) for child_contexts in self.child_context_sets]):

            # Apply candidate_filter if one was given
            # Accepts a tuple of Context objects (e.g., (Span, Span))
            # (candidate_filter returns whether or not proposed candidate passes throttling condition)
            if self.candidate_filter:
                if not self.candidate_filter(tuple(args[i][1] for i in range(self.arity))):
                    continue

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
            if not clear:
                q = select([self.candidate_class.id])
                for key, value in candidate_args.items():
                    q = q.where(getattr(self.candidate_class, key) == value)
                candidate_id = self.session.execute(q).first()
                if candidate_id is not None:
                    continue

            # Add Candidate to session
            yield self.candidate_class(**candidate_args)


class OmniNgrams(Ngrams):
    """
    Defines the space of candidates.

    Defines the space of candidates as all n-grams (n <= n_max) in a Document _x_,
    divided into Phrases inside of html elements (such as table cells).
    """

    def __init__(self, n_max=5, split_tokens=['-', '/']):
        """
        Initialize OmniNgrams.
        """
        Ngrams.__init__(self, n_max=n_max, split_tokens=split_tokens)

    def apply(self, session, context):
        """
        Generate OmniNgrams from a Document by parsing all of its Phrases.
        """
        if not isinstance(context, Document):
            raise TypeError("Input Contexts to OmniNgrams.apply() must be of type Document")

        doc = session.query(Document).filter(Document.id == context.id).one()
        for phrase in doc.phrases:
            for ts in Ngrams.apply(self, phrase):
                yield ts


class OmniFigures(CandidateSpace):
    """
    Defines the space of candidates as all figures in a Document _x_,
    indexing by **position offset**.
    """

    def __init__(self, type=None):
        """
        Initialize OmniFigures.

        Only support figure type filter.
        """
        CandidateSpace.__init__(self)
        if type is not None:
            self.type=type.strip().lower()
        self.type = None

    def apply(self, session, context):
        """
        Generate OmniFigures from a Document by parsing all of its Figures.
        """
        if not isinstance(context, Document):
            raise TypeError("Input Contexts to OmniFigures.apply() must be of type Document")

        doc = session.query(Document).filter(Document.id == context.id).one()
        for figure in doc.figures:
            if self.type is None or figure.url.lower().endswith(self.type):
                yield TemporaryImage(figure)
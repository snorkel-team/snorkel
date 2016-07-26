from itertools import chain
from multiprocessing import Process, Queue, JoinableQueue
from snorkel import SnorkelBase, SnorkelSession
from parser import Context
from sqlalchemy import Column, String, Integer, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.types import PickleType
from Queue import Empty


QUEUE_COLLECT_TIMEOUT = 5


class Candidates(SnorkelBase):
    """A named collection of Candidate objects."""
    __tablename__ = 'candidates'
    id = Column(String, primary_key=True)

    candidates = relationship('Candidate', backref='candidates')

    def __repr__(self):
        return "Candidates" + str((self.id,))

    def __iter__(self):
        """Default iterator is over Candidate objects"""
        for candidate in self.candidates:
            yield candidate

    def __len__(self):
        return len(self.candidates)

    def get_candidates(self):
        return self.candidates

    def get_candidate(self, candidate_id):
        """Retrieve a Candidate by id"""
        session = SnorkelSession.object_session(self)
        return session.query(Candidate).filter(Candidate.id == candidate_id).one()

    def get_candidates_in(self, context_id):
        """Return the candidates in a specific context (e.g. Sentence)"""
        session = SnorkelSession.object_session(self)
        return session.query(Candidate).filter(Context.id == context_id)

    def gold_stats(self, gold_set):
        """Return precision and recall relative to a "gold" set of candidates of the same type"""
        # TODO: Make this efficient via SQL
        gold = gold_set if isinstance(gold_set, set) else set(gold_set)
        cs   = self.get_candidates()
        nc   = len(cs)
        ng   = len(gold)
        both = len(gold.intersection(cs))
        print "# of gold annotations\t= %s" % ng
        print "# of candidates\t\t= %s" % nc
        print "Candidate recall\t= %0.3f" % (both / float(ng),)
        print "Candidate precision\t= %0.3f" % (both / float(nc),)


class Candidate(SnorkelBase):
    """
    A candidate k-arity relation, **uniquely identified by its id**.
    """
    __tablename__ = 'candidate'
    id = Column(String, primary_key=True)
    type = Column(String)
    candidates_id = Column(String, ForeignKey('candidates.id'))
    context_id = Column(String, ForeignKey('context.id'))

    __mapper_args__ = {
        'polymorphic_identity': 'candidate',
        'polymorphic_on': type
    }


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

    def extract(self, contexts):
        c = Candidates()

        if self.parallelism in [1, False]:
            for candidate in self._extract(contexts):
                c.candidates.append(candidate)
        else:
            for candidate in self._extract_multiprocess(contexts):
                c.candidates.append(candidate)

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


class Ngram(Candidate):
    """
    A span of _n_ tokens, identified by Context id and character-index start, end (inclusive).

    char_offsets are **relative to the Document start**
    """
    __tablename__ = 'ngram'
    id = Column(String, ForeignKey('candidate.id'), primary_key=True)
    char_start = Column(Integer)
    char_end = Column(Integer)
    meta = Column(PickleType)

    __mapper_args__ = {
        'polymorphic_identity': 'ngram',
    }

    def __len__(self):
        return self.char_end - self.char_start + 1

    def get_word_start(self):
        return self.char_to_word_index(self.char_start)

    def get_word_end(self):
        return self.char_to_word_index(self.char_end)

    def get_n(self):
        return self.get_word_end() - self.get_word_start() + 1

    def get_sent_offset(self):
        return self.context.char_offsets[0]

    def get_sent_char_start(self):
        return self.char_start - self.get_sent_offset()

    def get_sent_char_end(self):
        return self.char_end - self.get_sent_offset()

    def char_to_word_index(self, ci):
        """Given a character-level index (offset), return the index of the **word this char is in**"""
        i = None
        for i, co in enumerate(self.context.char_offsets):
            if ci == co:
                return i
            elif ci < co:
                return i-1
        return i

    def word_to_char_index(self, wi):
        """Given a word-level index, return the character-level index (offset) of the word's start"""
        return self.context.char_offsets[wi]

    def get_attrib_tokens(self, a):
        """Get the tokens of sentence attribute _a_ over the range defined by word_offset, n"""
        return self.context.__getattribute__(a)[self.get_word_start():self.get_word_end() + 1]

    def get_attrib_span(self, a, sep=" "):
        """Get the span of sentence attribute _a_ over the range defined by word_offset, n"""
        # NOTE: Special behavior for words currently (due to correspondence with char_offsets)
        if a == 'words':
            return self.context.text[self.get_sent_char_start():self.get_sent_char_end() + 1]
        else:
            return sep.join(self.get_attrib_tokens(a))

    def get_span(self, sep=" "):
        return self.get_attrib_span('words', sep)

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
            return Ngram(char_start, char_end, self.context)
        else:
            raise NotImplementedError()

    def __repr__(self):
        return 'Ngram("%s", id=%s, chars=[%s,%s], words=[%s,%s])' \
            % (" ".join(self.context.words), self.id, self.char_start, self.char_end, self.get_word_start(), self.get_word_end())


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
                ws = context.words[i:i+l]
                # NOTE that we derive char_len without using sep
                cl = context.char_offsets[i+l-1] - context.char_offsets[i] + len(context.words[i+l-1])
                char_start = context.char_offsets[i]
                char_end = context.char_offsets[i] + cl - 1
                ngram_id = "ngram-%s-%s-%s" % (context.id, str(char_start), str(char_end))
                yield Ngram(id=ngram_id, char_start=char_start, char_end=char_end, context=context)

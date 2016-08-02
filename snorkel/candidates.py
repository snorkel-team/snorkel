from .models import CandidateSet, Ngram
from itertools import chain
from multiprocessing import Process, Queue, JoinableQueue
from Queue import Empty
from utils import get_as_dict

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
        self.feats = None
        self.feat_index = {}
        # self.contexts = contexts

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


# # Basic sentence attributes
# WORDS        = 'words'
# CHAR_OFFSETS = 'char_offsets'
# TEXT         = 'text'

# class Ngram(Candidate):
#     """A span of _n_ tokens, identified by sentence id and character-index start, end (inclusive)"""
#     def __init__(self, char_start, char_end, sent, metadata={}):

#         # Inherit full sentence object (tranformed to dict) and check for necessary attribs
#         self.sentence = get_as_dict(sent)
#         self.sent_id  = self.sentence['id']
#         REQ_ATTRIBS = ['id', WORDS]
#         if not all([self.sentence.has_key(a) for a in REQ_ATTRIBS]):
#             raise ValueError("Sentence object must have attributes %s to form Ngram object" % ", ".join(REQ_ATTRIBS))

#         # Set basic object attributes
#         self.id          = "%s:%s-%s" % (self.sent_id, char_start, char_end)
#         self.char_start  = char_start
#         self.char_end    = char_end
#         self.char_len    = char_end - char_start + 1
#         self.word_start  = self.char_to_word_index(char_start)
#         self.word_end    = self.char_to_word_index(char_end)
#         self.n           = self.word_end - self.word_start + 1

#         # NOTE: We assume that the char_offsets are **relative to the document start**
#         self.sent_offset     = self.sentence[CHAR_OFFSETS][0]
#         self.sent_char_start = self.char_start - self.sent_offset
#         self.sent_char_end   = self.char_end - self.sent_offset

#         # A dictionary to hold task-specific metadata e.g. canonical id, category, etc.
#         self.metadata = metadata

#         # To enable generic methods
#         self.context_id = self.sent_id

#     def char_to_word_index(self, ci):
#         """Given a character-level index (offset), return the index of the **word this char is in**"""
#         for i,co in enumerate(self.sentence[CHAR_OFFSETS]):
#             if ci == co:
#                 return i
#             elif ci < co:
#                 return i-1
#         return i

#     def word_to_char_index(self, wi):
#         """Given a word-level index, return the character-level index (offset) of the word's start"""
#         return self.sentence[CHAR_OFFSETS][wi]

#     def get_attrib_tokens(self, a):
#         """Get the tokens of sentence attribute _a_ over the range defined by word_offset, n"""
#         return self.sentence[a][self.word_start:self.word_end+1]

#     def get_attrib_span(self, a, sep=" "):
#         """Get the span of sentence attribute _a_ over the range defined by word_offset, n"""
#         # NOTE: Special behavior for words currently (due to correspondence with char_offsets)
#         if a == WORDS:
#             return self.sentence[TEXT][self.sent_char_start:self.sent_char_end+1]
#         else:
#             return sep.join(self.get_attrib_tokens(a))

#     def get_span(self, sep=" "):
#         return self.get_attrib_span(WORDS)

#     def __getitem__(self, key):
#         """
#         Slice operation returns a new candidate sliced according to **char index**
#         Note that the slicing is w.r.t. the candidate range (not the abs. sentence char indexing)
#         """
#         if isinstance(key, slice):
#             char_start = self.char_start if key.start is None else self.char_start + key.start
#             if key.stop is None:
#                 char_end = self.char_end
#             elif key.stop >= 0:
#                 char_end = self.char_start + key.stop - 1
#             else:
#                 char_end = self.char_end + key.stop
#             return Ngram(char_start, char_end, self.sentence)
#         else:
#             raise NotImplementedError()

#     def __repr__(self):
#         return '<Ngram("%s", id=%s, chars=[%s,%s], words=[%s,%s])' \
#             % (self.get_attrib_span(WORDS), self.id, self.char_start, self.char_end, self.word_start, self.word_end)

# >>>>>>> learning-refactor

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
                yield Ngram(char_start=char_start, char_end=char_end, context=context)

                # Check for split
                # NOTE: For simplicity, we only split single tokens right now!
                if l == 1 and self.split_rgx is not None:
                    m = re.search(self.split_rgx,
                        text[char_start-context.char_offsets[0]:char_end-context.char_offsets[0]+1])
                    if m is not None and l < self.n_max:
                        yield Ngram(char_start=char_start, char_end=char_start + m.start(1) - 1, sent=s)
                        yield Ngram(char_start=char_start + m.end(1), char_end=char_end, sent=s)

class TableNgrams(Ngrams):
    """
    Defines the space of candidates as all n-grams (n <= n_max) in a Table _x_,
    indexing by **character offset**.
    """
    def apply(self, context):
        try:
            phrases = context.phrases
        except:
            phrases = [context]
            # raise ValueError("Input object must have %s attribute" % 'phrases')

        for phrase in phrases:
            for ngram in super(TableNgrams, self).apply(phrase):
                yield ngram


# """-------------------------HERE BE BRADEN'S KINGDOM-------------------------"""

# class TableNgram(Ngram):
#     def __init__(self, phrase, ngram, table):
#         super(TableNgram, self).__init__(ngram.char_start, ngram.char_end, ngram.sentence)
#         self.context_id = phrase.context_id
#         self.table_id = phrase.table_id
#         self.cell_id = phrase.cell_id
#         self.row_num = phrase.row_num
#         self.col_num = phrase.col_num
#         self.html_tag = phrase.html_tag
#         self.html_attrs = phrase.html_attrs
#         self.html_anc_tags = phrase.html_anc_tags
#         self.html_anc_attrs = phrase.html_anc_attrs
#         self.context = table

#     def __repr__(self):
#         return '<TableNgram("%s", id=%s, chars=[%s,%s], (row,col)=(%s,%s), tag=%s)' \
#             % (self.get_attrib_span(WORDS), self.id, self.char_start, self.char_end, self.row_num, self.col_num, self.html_tag)

# ### HACKY ###
#     def aligned(self, attribute='words', case_sensitive=False):
#         return (self.row_window(case_sensitive=case_sensitive)
#               + self.col_window(case_sensitive=case_sensitive))

#     def row_window(self, attribute='words', case_sensitive=False):
#         ngrams = [ngram for ngram in self.get_aligned_ngrams(self.context, axis='row')]
#         if not case_sensitive:
#             return [ngram.lower() for ngram in ngrams]
#         else:
#             return ngrams

#     def col_window(self, attribute='words', case_sensitive=False):
#         ngrams = [ngram for ngram in self.get_aligned_ngrams(self.context, axis='col')]
#         if not case_sensitive:
#             return [ngram.lower() for ngram in ngrams]
#         else:
#             return ngrams

#     # NOTE: it may just be simpler to search by row_num, col_num rather than
#     # traversing tree, though other range features may benefit from tree structure
#     def get_aligned_ngrams(self, context, n_max=3, attribute='words', axis='row'):
#         # SQL join method (eventually)
#         if axis=='row':
#             phrase_ids = [phrase.id for phrase in context.phrases.values() if phrase.row_num == self.row_num]
#         elif axis=='col':
#             phrase_ids = [phrase.id for phrase in context.phrases.values() if phrase.col_num == self.col_num]
#         for phrase_id in phrase_ids:
#             words = context.phrases[phrase_id].words
#             for ngram in self.get_ngrams(words, n_max=n_max):
#                 yield ngram
#         # Tree traversal method:
#         # root = et.fromstring(context.html)
#         # if axis=='row':
#             # snorkel_ids = root.xpath('//*[@snorkel_id="%s"]/following-sibling::*/@snorkel_id' % cand.cell_id)
#         # if axis=='col':
#             # position = len(root.xpath('//*[@snorkel_id="%s"]/following-sibling::*/@snorkel_id' % cand.cell_id)) + 1
#             # snorkel_ids = root.xpath('//*[@snorkel_id][position()=%d]/@snorkel_id' % position)

#     # replace with a library function?
#     def get_ngrams(self, words, n_max=3):
#         N = len(words)
#         for root in range(N):
#             for n in range(min(n_max, N - root)):
#                 yield '_'.join(words[root:root+n+1])


# class TableNgrams(Ngrams):
#     """
#     Defines the space of candidates as all n-grams (n <= n_max) in a cell within a table _x_
#     "Calling _apply(x)_ given an object _x_ returns a generator over candidates in _x_."
#     """
#     def apply(self, context):
#         # table = context if isinstance(context, dict) else context._asdict()
#         try:
#             phrases = context.phrases
#         except:
#             raise ValueError("Input object must have %s attribute" % 'phrases')

#         for phrase in phrases.values():
#             for ngram in super(TableNgrams, self).apply(phrase):
#                 yield TableNgram(phrase, ngram, context)


# >>>>>>> tables

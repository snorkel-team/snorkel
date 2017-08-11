from snorkel.contrib.babble import Explanation

def get_user_lists():
    return {
        'colors':['red','green','blue'],
        'bluebird':['blue','bird','fly'],
        'greek':['alpha','beta','gamma'],
        'letters':['a','B','C'],
        'smalls':['a','b','c','d'],
        'luckies': [7, 8, 9],
        'unluckies': [0, 13, 66],
    }

# Test candidate (hash: 668761641257950361):
# "City land records show that GM President [Daniel Ammann] and his wife, 
# [Pernilla Ammann], bought the 15-bedroom mansion on Balmoral Drive in 
# the upscale historic neighborhood on July 31."

logic = [
    # Base
    Explanation(
        condition="True",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # And
    Explanation(
        condition="True and True",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # Or
    Explanation(
        condition="False or True",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # Not boolean
    Explanation(
        condition="not False",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # Not function
    Explanation(
        condition="2 is not less than 1",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # All
    Explanation(
        condition='all of (2, 3, 4) are greater than 1',
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # Any
    Explanation(
        condition='any of (3, 1, 4) are less than 2',
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # None
    Explanation(
        condition='none of (1, 2, 3) are greater than 4',
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),                
]

grouping = [
    # Parentheses
    Explanation(
        condition="True or (True and False)",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
]

integers = [
    # Equals (Int)
    Explanation(
        condition="1 is equal to 1",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # Integers (digit or text)
    Explanation(
        condition="1 is equal to one",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # Less than
    Explanation(
        condition="1 is less than 2",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # At most
    Explanation(
        condition="2 is less than or equal to 2",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # Greater than
    Explanation(
        condition="2 > 1",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # At least
    Explanation(
        condition="2 is at least 2",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),    
]
lists = [
    # # In
    # Explanation(
    #     condition="1 is in (1, 2)",
    #     label=True,
    #     candidate=('foo', 'bar'),
    #     semantics=None),
    # # Contains
    # Explanation(
    #     condition="(1, 2) contains 2",
    #     label=True,
    #     candidate=('foo', 'bar'),
    #     semantics=None),
    # # List
    # Explanation(
    #     condition="1 equals 2, 1, or 3",
    #     label=True,
    #     candidate=('foo', 'bar'),
    #     semantics=None),
    # UserList
    Explanation(
        condition="7 is in the luckies",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # # OrList left
    # Explanation(
    #     condition="7 or 6 is in the luckies",
    #     label=True,
    #     candidate=('foo', 'bar'),
    #     semantics=None),
    # # OrList right
    # Explanation(
    #     condition="2 is less than 3 or 1",
    #     label=True,
    #     candidate=('foo', 'bar'),
    #     semantics=None),
    # # AndList left
    # Explanation(
    #     condition="7 and 8 are in the luckies",
    #     label=True,
    #     candidate=('foo', 'bar'),
    #     semantics=None),
    # # AndList right
    # Explanation(
    #     condition="2 is less than 3 and 4",
    #     label=True,
    #     candidate=('foo', 'bar'),
    #     semantics=None),    
]

absorption = [
    # Partially unparseable
    Explanation(
        condition="1 is less than 2 and the moon is full",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None)
]


explanations = (logic + grouping + integers + lists + absorption)

# TODO: re-add the following:
#     # # Index OrList right
#     # Explanation(
#     #     condition="'wife' is less than three words to the left of arg 1 or arg2",
#     #     candidate=668761641257950361,
#     #     denotation=1,
#     # semantics=None),
#     # # Index within
#     # Explanation(
#     #     condition="'wife' is within three words of arg 1",
#     #     candidate=668761641257950361,
#     #     denotation=1,
#     # semantics=None),
#     # # Index without
#     # Explanation(
#     #     condition="arg 1 is not within 5 words of arg 2",
#     #     candidate=668761641257950361,
#     #     denotation=1,
#     # semantics=None),

    # # Intersection0
    # Explanation(
    #     condition="there is at least one word from colors in the bluebird words",
    #     candidate=('foo', 'bar'),
    #     denotation=1,
    # semantics=None),
    # # Intersection1
    # Explanation(
    #     condition="less than two colors words are in bluebird",
    #     candidate=('foo', 'bar'),
    #     denotation=1,
    # semantics=None),
    # # Disjoint
    # Explanation(
    #     condition="there are no colors words in the greek words",
    #     candidate=('foo', 'bar'),
    #     denotation=1,
    # semantics=None),

    # #     # Composition0
# #     Explanation(
# #         condition="'wife' is between arg 1 and arg 2 and 'years' is to the left of arg 1",
        # # label=True,
# #         candidate=668761641257950361,
# # semantics=None),
# #     # Composition1
# #     Explanation(
# #         condition="arg 1 is identical to arg 2",
        # # label=True,
# #         candidate=('foo', 'foo'),
# # semantics=None),
# #     # Composition2
# #     Explanation(
# #         condition="there is at least one spouse word between arg 1 and arg 2",
        # # label=True,
# #         candidate=668761641257950361,
# # semantics=None),
# #     # Composition3
# #     Explanation(
# #         condition="there is at least one spouse word within two words to the left of arg 1 or arg 2",
        # # label=True,
# #         candidate=668761641257950361,
# # semantics=None),


## To generate a small database with the expected candidate:
# from snorkel.models import Document, candidate_subclass
# from snorkel.parser import TSVDocPreprocessor, CorpusParser
# from snorkel.parser.spacy_parser import Spacy
# from snorkel.candidates import Ngrams, CandidateExtractor
# from snorkel.matchers import PersonMatcher
# from snorkel.contrib.babble import SemanticParser, Explanation

# test_article_path = os.environ['SNORKELHOME'] + '/test/babble/test_article.tsv'
# doc_preprocessor = TSVDocPreprocessor(test_article_path)
# corpus_parser = CorpusParser(parser=Spacy())
# corpus_parser.apply(doc_preprocessor)
# Spouse = candidate_subclass('Spouse', ['person1', 'person2'])
# ngrams         = Ngrams(n_max=2)
# person_matcher = PersonMatcher(longest_match_only=True)
# cand_extractor = CandidateExtractor(Spouse, [ngrams, ngrams], [person_matcher, person_matcher], symmetric_relations=True)
# docs = session.query(Document).order_by(Document.name).all()
# sents = [s for doc in docs for s in doc.sentences]
# cand_extractor.apply(sents, split=0)
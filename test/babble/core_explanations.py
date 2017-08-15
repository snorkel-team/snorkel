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
        semantics=('.root', ('.label', ('.bool', True), ('.bool', True)))),
    # And
    Explanation(
        condition="True and True",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.and', ('.bool', True), ('.bool', True))))),
    # Or
    Explanation(
        condition="False or True",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.or', ('.bool', False), ('.bool', True))))),
    # Not boolean
    Explanation(
        condition="not False",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.not', ('.bool', False))))),
    # Not function
    Explanation(
        condition="2 is not less than 1",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.not', ('.call', ('.lt', ('.int', 1)), ('.int', 2)))))),
    # All
    Explanation(
        condition='all of (2, 3, 4) are greater than 1',
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.all', ('.map', ('.gt', ('.int', 1)), ('.list', ('.int', 2), ('.int', 3), ('.int', 4))))))),
    # Any
    Explanation(
        condition='any of (3, 1, 4) are less than 2',
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.any', ('.map', ('.lt', ('.int', 2)), ('.list', ('.int', 3), ('.int', 1), ('.int', 4))))))),
    # None
    Explanation(
        condition='none of (1, 2, 3) are greater than 4',
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.none', ('.map', ('.gt', ('.int', 4)), ('.list', ('.int', 1), ('.int', 2), ('.int', 3))))))),                
]

grouping = [
    # Parentheses
    Explanation(
        condition="True or (True and False)",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.or', ('.bool', True), ('.and', ('.bool', True), ('.bool', False)))))),
]

integers = [
    # Equals (Int)
    Explanation(
        condition="1 is equal to 1",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.eq', ('.int', 1)), ('.int', 1))))),
    # Integers (digit or text)
    Explanation(
        condition="1 is equal to one",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.eq', ('.int', 1)), ('.int', 1))))),
    # Less than
    Explanation(
        condition="1 is less than 2",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.lt', ('.int', 2)), ('.int', 1))))),
    # At most
    Explanation(
        condition="2 is less than or equal to 2",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.leq', ('.int', 2)), ('.int', 2))))),
    # Greater than
    Explanation(
        condition="2 > 1",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.gt', ('.int', 1)), ('.int', 2))))),
    # At least
    Explanation(
        condition="2 is at least 2",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.geq', ('.int', 2)), ('.int', 2))))),    
]

lists = [
    # OrList left
    Explanation(
        condition="7 or 5 is larger than 6",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root',('.label',('.bool', True),('.any',('.map', ('.gt', ('.int', 6)), ('.list', ('.int', 7), ('.int', 5))))))),
    # OrList right
    Explanation(
        condition="2 is less than 3 or 1",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root',('.label',('.bool', True),('.call',('.composite_or', ('.lt',), ('.list', ('.int', 3), ('.int', 1))),('.int', 2))))),
    # AndList left
    Explanation(
        condition="8 and 8 are equal to 8",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root',('.label',('.bool', True),('.all',('.map', ('.eq', ('.int', 8)), ('.list', ('.int', 8), ('.int', 8))))))),
    # AndList right
    Explanation(
        condition="2 is less than 3 and 4",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root',('.label',('.bool', True),('.call',('.composite_and', ('.lt',), ('.list', ('.int', 3), ('.int', 4))),('.int', 2))))),
    # Not AndList
    Explanation(
        condition="2 is not more than 1 and 3",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root',('.label',('.bool', True),('.not',('.call',('.composite_and', ('.gt',), ('.list', ('.int', 1), ('.int', 3))),('.int', 2)))))),
    # Not OrList
    Explanation(
        condition="2 is not more than 3 or 4",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root',('.label',('.bool', True),('.not',('.call',('.composite_or', ('.gt',), ('.list', ('.int', 3), ('.int', 4))),('.int', 2)))))),
]

membership = [
    # In
    Explanation(
        condition="1 is in (1, 2)",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.in', ('.list', ('.int', 1), ('.int', 2))), ('.int', 1))))),
    # In AndList
    Explanation(
        condition="1 and 2 are in (1, 2, 3)",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.all', ('.map', ('.in', ('.list', ('.int', 1), ('.int', 2), ('.int', 3))), ('.list', ('.int', 1), ('.int', 2))))))),
    # In OrList
    Explanation(
        condition="1 or 2 is in (2, 3)",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.any', ('.map', ('.in', ('.list', ('.int', 2), ('.int', 3))), ('.list', ('.int', 1), ('.int', 2))))))),
    # Contains
    Explanation(
        condition="(1, 2) contains 2",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.in', ('.list', ('.int', 1), ('.int', 2))), ('.int', 2))))),
    # Contains AndList
    Explanation(
        condition="(1, 2) contains 2 and 1",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.all', ('.map', ('.in', ('.list', ('.int', 1), ('.int', 2))), ('.list', ('.int', 2), ('.int', 1))))))),
    # Contains OrList
    Explanation(
        condition="(1, 2) contains 2 or 3",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.any', ('.map', ('.in', ('.list', ('.int', 1), ('.int', 2))), ('.list', ('.int', 2), ('.int', 3))))))),
]

absorption = [
    # Partially unparseable
    Explanation(
        condition="1 is less than 2 and the moon is full",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.lt', ('.int', 2)), ('.int', 1)))))
]


explanations = (logic + grouping + integers + lists + membership + absorption)

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
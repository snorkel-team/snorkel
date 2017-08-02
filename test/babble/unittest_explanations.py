from snorkel.contrib.babble import Explanation

def get_user_lists():
    return {
        'colors':['red','green','blue'],
        'bluebird':['blue','bird','fly'],
        'greek':['alpha','beta','gamma'],
        'letters':['a','B','C'],
        'smalls':['a','b','c','d'],
        'spouse':['wife','husband','spouse']}

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
        condition="'blue' is not in all caps",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # All
    Explanation(
        condition='all of the colors are lowercase',
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # Any
    Explanation(
        condition='any of the letters are lowercase',
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # None
    Explanation(
        condition='none of the smalls are capitalized',
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

strings = [
    # Equals (String)
    Explanation(
        condition="'yes' equals 'yes'",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # Lowercase
    Explanation(
        condition="arg 1 is lowercase",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # Uppercase
    Explanation(
        condition="arg 1 is upper case",
        label=True,
        candidate=('FOO', 'bar'),
        semantics=None),
    # Capitalized
    Explanation(
        condition="arg 1 is capitalized",
        label=True,
        candidate=('Foo', 'bar'),
        semantics=None),
    # Starts with
    Explanation(
        condition="the word 'blueberry' starts with 'blue'",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # Ends with
    Explanation(
        condition="the word 'blueberry' ends with 'berry'",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
]

lists = [
    # In
    Explanation(
        condition="'bar' is in 'foobarbaz'",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # Contains
    Explanation(
        condition="the word 'foobarbaz' contains 'oobarba'",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # List
    Explanation(
        condition="'bar' equals 'foo', 'bar', or 'baz'",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # UserList
    Explanation(
        condition="'blue' in colors",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # OrList left
    Explanation(
        condition="'blue' or 'shmoo' is in colors",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # OrList right
    Explanation(
        condition="'blue' ends with 'moe' or 'lue'",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # AndList left
    Explanation(
        condition="'blue' and 'red' are in colors",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # AndList right
    Explanation(
        condition="'blue' contains 'l' and 'u'",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),    
]

candidate_helpers = [
    # Candidate as string
    Explanation(
        condition="argument 1 is 'foo'",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None),
    # Left words (list)
    Explanation(
        condition="'wife' is in the words left of arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None), 
    # Right words (list)
    Explanation(
        condition="'wife' is in the words to the right of arg 1",
        label=True,
        candidate=668761641257950361,
        semantics=None), 
    # Between words (list)
    Explanation(
        condition="'wife' is in the words between arg 1 and arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None), 
    # Sentence (list)
    Explanation(
        condition='"wife" is in the sentence',
        label=True,
        candidate=668761641257950361,
        semantics=None),    
]

index_comparisons = [
    # Index left
    Explanation(
        condition="arg 1 is left of arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Index right
    Explanation(
        condition="arg 2 is right of arg 1",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Between
    Explanation(
        condition="'wife' is between arg 1 and arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Index left equality
    Explanation(
        condition="'wife' is two words to the left of arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Index left inequality 0
    Explanation(
        condition="arg 1 is more than three words to the left of arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Index left inequality 1
    Explanation(
        condition="not arg 1 is more than fifty words to the left of arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Index left inequality 2
    Explanation(
        condition="',' is immediately to the left of arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Index left inequality 3
    Explanation(
        condition="',' is right before arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),                      
    # Index within (<=)
    Explanation(
        condition="'wife' is within three words to the left of arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Index OrList left
    Explanation(
        condition="'husband' or 'wife' is within three words to the left of arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Index AndList left
    Explanation(
        condition="not 'husband' and 'wife' are within three words to the left of arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Characters0
    Explanation(
        condition="'wife' is less than 10 characters to the left of arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Characters1
    Explanation(
        condition="'wife' is more than 5 characters to the right of arg 1",
        label=True,
        candidate=668761641257950361,
        semantics=None),       
]

pos_ner = [
    # Tokens
    Explanation(
        condition="at least one word to the left of arg 1 is lower case",
        label=True,
        candidate=668761641257950361,
        semantics=None), 
    # POS
    Explanation(
        condition="at least one noun exists between arg 1 and arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # NER
    Explanation(
        condition="there are no people between arg 1 and arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),     
]

count = [
    # Count0
    Explanation(
        condition="there are not three people in the sentence",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Count1
    Explanation(
        condition="the number of words between arg 1 and arg 2 is less than 25",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Count2
    Explanation(
        condition="there are more than 3 words between arg 1 and arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Count3
    Explanation(
        condition="at least one word exists between arg 1 and arg 2",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Count4
    Explanation(
        condition="there are two nouns to the left of arg 1",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Count5
    Explanation(
        condition="there are less than three nouns to the left of arg 1",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Count6
    Explanation(
        condition="there are not more than two nouns to the left of arg 1",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # Count7
    Explanation(
        condition="at least one word to the left of arg 2 starts with a spouse word",
        label=True,
        candidate=668761641257950361,
        semantics=None),    
]

absorption = [
    # Partially unparseable
    Explanation(
        condition="1 is less than 2 and the moon is full",
        label=True,
        candidate=('foo', 'bar'),
        semantics=None)
]

anaphora = [
    # Them
    Explanation(
        condition="'wife' is between arg 1 and arg 2 and 'divorced' is not between them",
        label=True,
        candidate=668761641257950361,
        semantics=None),
    # TODO: he/she, his/her, him/her
]

inversion = [
    # Inverted sentence
    Explanation(
        condition="to the left of arg 2 is a spouse word",
        label=True,
        candidate=668761641257950361,
        semantics=None),
]

tuples = [
    # Tuple
    Explanation(
        condition="the pair (arg 1, arg 2) is the same as the tuple ('foo', 'bar')",
        label=True,
        candidate=('foo', 'bar'),
        semantics=('.root', ('.label', ('.bool', True), ('.call', ('.eq', ('.tuple', ('.list', ('.string', u'foo'), ('.string', u'bar')))), ('.tuple', ('.list', ('.arg_to_string', ('.arg', ('.int', 1))), ('.arg_to_string', ('.arg', ('.int', 2))))))))),
]

explanations = (logic + grouping + integers + strings + lists + candidate_helpers + 
            index_comparisons + pos_ner + count + absorption + anaphora + 
            inversion + tuples)

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
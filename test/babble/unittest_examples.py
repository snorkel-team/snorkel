from snorkel.contrib.babble import Example

user_lists = {
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
    Example(
        explanation="label True because True",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # And
    Example(
        explanation="label True because True and True",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # Or
    Example(
        explanation="label True because False or True",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # Not boolean
    Example(
        explanation="label True because not False",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # Not function
    Example(
        explanation="label True because 'blue' is not in all caps",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # All
    Example(
        explanation='label True because all of the colors are lowercase',
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # Any
    Example(
        explanation='label True because any of the letters are lowercase',
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # None
    Example(
        explanation='label True because none of the smalls are capitalized',
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),                
]

grouping = [
    # Parentheses
    Example(
        explanation="label True because True or (True and False)",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
]

integers = [
    # Equals (Int)
    Example(
        explanation="label True because 1 is equal to 1",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # Integers (digit or text)
    Example(
        explanation="label True because 1 is equal to one",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # Less than
    Example(
        explanation="label True because 1 is less than 2",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # At most
    Example(
        explanation="label True because 2 is less than or equal to 2",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # Greater than
    Example(
        explanation="label True because 2 > 1",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # At least
    Example(
        explanation="label True because 2 is at least 2",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),    
]

strings = [
    # Equals (String)
    Example(
        explanation="label True because 'yes' equals 'yes'",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # Lowercase
    Example(
        explanation="label True because arg 1 is lowercase",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # Uppercase
    Example(
        explanation="label True because arg 1 is upper case",
        candidate=('FOO', 'bar'),
        denotation=1,
        semantics=None),
    # Capitalized
    Example(
        explanation="label True because arg 1 is capitalized",
        candidate=('Foo', 'bar'),
        denotation=1,
        semantics=None),
    # Starts with
    Example(
        explanation="label True because the word 'blueberry' starts with 'blue'",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # Ends with
    Example(
        explanation="label True because the word 'blueberry' ends with 'berry'",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
]

lists = [
    # In
    Example(
        explanation="label True because 'bar' is in 'foobarbaz'",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # Contains
    Example(
        explanation="label True because the word 'foobarbaz' contains 'oobarba'",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # List
    Example(
        explanation="label True because 'bar' equals 'foo', 'bar', or 'baz'",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # UserList
    Example(
        explanation="label True because 'blue' in colors",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # OrList left
    Example(
        explanation="label True because 'blue' or 'shmoo' is in colors",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # OrList right
    Example(
        explanation="label True because 'blue' ends with 'moe' or 'lue'",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # AndList left
    Example(
        explanation="label True because 'blue' and 'red' are in colors",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # AndList right
    Example(
        explanation="label True because 'blue' contains 'l' and 'u'",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),    
]

candidate_helpers = [
    # Candidate as string
    Example(
        explanation="label True because argument 1 is 'foo'",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None),
    # Left words (list)
    Example(
        explanation="label True because 'wife' is in the words left of arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None), 
    # Right words (list)
    Example(
        explanation="label True because 'wife' is in the words to the right of arg 1",
        candidate=668761641257950361,
        denotation=1,
        semantics=None), 
    # Between words (list)
    Example(
        explanation="label True because 'wife' is in the words between arg 1 and arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None), 
    # Sentence (list)
    Example(
        explanation='label True because "wife" is in the sentence',
        candidate=668761641257950361,
        denotation=1,
        semantics=None),    
]

index_comparisons = [
    # Index left
    Example(
        explanation="label True because arg 1 is left of arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Index right
    Example(
        explanation="label True because arg 2 is right of arg 1",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Between
    Example(
        explanation="label True because 'wife' is between arg 1 and arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Index left equality
    Example(
        explanation="label True because 'wife' is two words to the left of arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Index left inequality 0
    Example(
        explanation="label True because arg 1 is more than three words to the left of arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Index left inequality 1
    Example(
        explanation="label True because not arg 1 is more than fifty words to the left of arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Index left inequality 2
    Example(
        explanation="label True because ',' is immediately to the left of arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Index left inequality 3
    Example(
        explanation="label True because ',' is right before arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),                      
    # Index within (<=)
    Example(
        explanation="label True because 'wife' is within three words to the left of arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Index OrList left
    Example(
        explanation="label True because 'husband' or 'wife' is within three words to the left of arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Index AndList left
    Example(
        explanation="label True because not 'husband' and 'wife' are within three words to the left of arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Characters0
    Example(
        explanation="label True because 'wife' is less than 10 characters to the left of arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Characters1
    Example(
        explanation="label True because 'wife' is more than 5 characters to the right of arg 1",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),       
]

pos_ner = [
    # Tokens
    Example(
        explanation="label True because at least one word to the left of arg 1 is lower case",
        candidate=668761641257950361,
        denotation=1,
        semantics=None), 
    # POS
    Example(
        explanation="label True because at least one noun exists between arg 1 and arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # NER
    Example(
        explanation="label True because there are no people between arg 1 and arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),     
]

count = [
    # Count0
    Example(
        explanation="label True because there are not three people in the sentence",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Count1
    Example(
        explanation="label True because the number of words between arg 1 and arg 2 is less than 25",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Count2
    Example(
        explanation="label True because there are more than 3 words between arg 1 and arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Count3
    Example(
        explanation="label True because at least one word exists between arg 1 and arg 2",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Count4
    Example(
        explanation="label True because there are two nouns to the left of arg 1",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Count5
    Example(
        explanation="label True because there are less than three nouns to the left of arg 1",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Count6
    Example(
        explanation="label True because there are not more than two nouns to the left of arg 1",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # Count7
    Example(
        explanation="label True because at least one word to the left of arg 2 starts with a spouse word",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),    
]

absorption = [
    # Partially unparseable
    Example(
        explanation="label True because 1 is less than 2 and the moon is full",
        candidate=('foo', 'bar'),
        denotation=1,
        semantics=None)
]

anaphora = [
    # Them
    Example(
        explanation="label True because 'wife' is between arg 1 and arg 2 and 'divorced' is not between them",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
    # TODO: he/she, his/her, him/her
]

inversion = [
    # Inverted sentence
    Example(
        explanation="label True because to the left of arg 2 is a spouse word",
        candidate=668761641257950361,
        denotation=1,
        semantics=None),
]

examples = (logic + grouping + integers + strings + lists + candidate_helpers + 
            index_comparisons + pos_ner + count + absorption + anaphora + 
            inversion)

# TODO: re-add the following:
#     # # Index OrList right
#     # Example(
#     #     explanation="label True because 'wife' is less than three words to the left of arg 1 or arg2",
#     #     candidate=668761641257950361,
#     #     denotation=1,
#     # semantics=None),
#     # # Index within
#     # Example(
#     #     explanation="label True because 'wife' is within three words of arg 1",
#     #     candidate=668761641257950361,
#     #     denotation=1,
#     # semantics=None),
#     # # Index without
#     # Example(
#     #     explanation="label True because arg 1 is not within 5 words of arg 2",
#     #     candidate=668761641257950361,
#     #     denotation=1,
#     # semantics=None),

    # # Intersection0
    # Example(
    #     explanation="label True because there is at least one word from colors in the bluebird words",
    #     candidate=('foo', 'bar'),
    #     denotation=1,
    # semantics=None),
    # # Intersection1
    # Example(
    #     explanation="label True because less than two colors words are in bluebird",
    #     candidate=('foo', 'bar'),
    #     denotation=1,
    # semantics=None),
    # # Disjoint
    # Example(
    #     explanation="label True because there are no colors words in the greek words",
    #     candidate=('foo', 'bar'),
    #     denotation=1,
    # semantics=None),

    # #     # Composition0
# #     Example(
# #         explanation="label True because 'wife' is between arg 1 and arg 2 and 'years' is to the left of arg 1",
# #         candidate=668761641257950361,
# #         denotation=1,
# # semantics=None),
# #     # Composition1
# #     Example(
# #         explanation="label True because arg 1 is identical to arg 2",
# #         candidate=('foo', 'foo'),
# #         denotation=1,
# # semantics=None),
# #     # Composition2
# #     Example(
# #         explanation="label True because there is at least one spouse word between arg 1 and arg 2",
# #         candidate=668761641257950361,
# #         denotation=1,
# # semantics=None),
# #     # Composition3
# #     Example(
# #         explanation="label True because there is at least one spouse word within two words to the left of arg 1 or arg 2",
# #         candidate=668761641257950361,
# #         denotation=1,
# # semantics=None),


## To generate a small database with the expected candidate:
# from snorkel.models import Document, candidate_subclass
# from snorkel.parser import TSVDocPreprocessor, CorpusParser
# from snorkel.parser.spacy_parser import Spacy
# from snorkel.candidates import Ngrams, CandidateExtractor
# from snorkel.matchers import PersonMatcher
# from snorkel.contrib.babble import SemanticParser, Example

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
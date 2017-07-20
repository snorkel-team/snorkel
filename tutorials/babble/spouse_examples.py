from snorkel.contrib.babble import Example

user_lists = {
    'spouse':['wife','husband','spouse'],
    'family':['father', 'mother', 'brother', 'sister']
} 

examples = [
#     explanations = [
#     "Label false because the number of words between arg 1 and arg 2 is larger than 10",
#     "Label false because there is a person between arg 1 and arg 2",
#     "Label true because there is at least one spouse word in the words between arg 1 and arg 2",
#     "Label true because there is at least one spouse word within two words to the left of arg 1 or arg 2",
#     "Label false because there are no spouse words in the sentence",
#     "Label true because the word 'and' is between arg 1 and arg 2 and 'married' is to the right of arg 2",
#     "Label false because there is at least one family word between arg 1 and arg 2",
#     "Label false because there is at least one family word within two words to the left of arg 1 or arg 2",
#     "Label false because there is at least one coworker word between arg 1 and arg 2",
#     "Label false because arg 1 is identical to arg 2",
#     ]
    Example(
        explanation="Label false because the number of words between arg 1 and arg 2 is larger than 10",
        candidate=-5729816328165410632,
        denotation=-1,
        semantics=None),
    Example(
        explanation="Label false because there is a person between arg 1 and arg 2",
        candidate=-8692729291220282012,
        denotation=-1,
        semantics=None),
    Example(
        explanation="Label true because there is at least one spouse word in the words between arg 1 and arg 2",
        candidate=-3135315734051751361,
        denotation=1,
        semantics=None),
    Example(
        explanation="Label true because there is at least one spouse word within two words to the left of arg 1 or arg 2",
        candidate=-7563346943193853808,
        denotation=1,
        semantics=None),
    Example(
        explanation="Label false because there are no spouse words in the sentence",
        candidate=-8021416815354059709,
        denotation=-1,
        semantics=None),
    Example(
        explanation="Label true because the word 'and' is between arg 1 and arg 2 and 'married' is in the sentence",
        candidate=None,
        denotation=1,
        semantics=None),
    Example(
        explanation="Label false because there are no spouse words in the sentence",
        candidate=-8021416815354059709,
        denotation=-1,
        semantics=None),
    Example(
        explanation="Label false because there is at least one family word between arg 1 and arg 2",
        candidate=-8692729291220282012,
        denotation=-1,
        semantics=None),
    Example(
        explanation="Label false because arg 1 is identical to arg 2",
        candidate=660552142898381681,
        denotation=-1,
        semantics=None),
]

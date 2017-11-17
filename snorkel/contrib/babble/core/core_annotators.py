import re

from ..grammar import Annotator

class PunctuationAnnotator(Annotator):
    def annotate(self, tokens):
        if len(tokens) == 1:
            if tokens[0]['pos'] in ["``", "\'\'"] or tokens[0]['word'] in ["'", '"']:
                return [('$Quote', tokens[0]['word'])]
            elif tokens[0]['pos'] == "-LRB-":
                return [('$OpenParen', tokens[0]['word'])]
            elif tokens[0]['pos'] == "-RRB-":
                return [('$CloseParen', tokens[0]['word'])]
        return []

class IntegerAnnotator(Annotator):
    def annotate(self, tokens):
        if len(tokens) == 1:
            value = None
            if tokens[0]['pos'] == 'CD':
                try:
                    token = tokens[0]['word']
                    value = int(float(token))
                except ValueError:
                    pass
            if value is None:
                try:
                    value = text2int(tokens[0]['word'])
                except:
                    pass
            if value is not None:
                return [('$Int', ('.int', value))]
        return []

# Deprecated: CoreNLP implementation
# class IntegerAnnotator(Annotator):
#     def annotate(self, tokens):
#         if len(tokens) == 1:
#             if all('normalizedNER' in token for token in tokens):
#                 ner_number = tokens[0]['normalizedNER']
#                 number = re.sub('[^\d\.]','', ner_number)
#                 value = int(float(number))
#                 return [('$Int', ('.int', value))]
#         return []

annotators = [PunctuationAnnotator(), IntegerAnnotator()]


def text2int(textnum, numwords={}):
    if not numwords:
        units = [
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
        "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
        "sixteen", "seventeen", "eighteen", "nineteen",
        ]

        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

        scales = ["hundred", "thousand", "million", "billion", "trillion"]

        numwords["and"] = (1, 0)
        for idx, word in enumerate(units):  numwords[word] = (1, idx)
        for idx, word in enumerate(tens):       numwords[word] = (1, idx * 10)
        for idx, word in enumerate(scales): numwords[word] = (10 ** (idx * 3 or 2), 0)

    ordinal_words = {'first':1, 'second':2, 'third':3, 'fifth':5, 'eighth':8, 'ninth':9, 'twelfth':12}
    ordinal_endings = [('ieth', 'y'), ('th', '')]

    textnum = textnum.replace('-', ' ')

    current = result = 0
    for word in textnum.split():
        if word in ordinal_words:
            scale, increment = (1, ordinal_words[word])
        else:
            for ending, replacement in ordinal_endings:
                if word.endswith(ending):
                    word = "%s%s" % (word[:-len(ending)], replacement)

            if word not in numwords:
                raise Exception("Illegal word: " + word)

            scale, increment = numwords[word]

        current = current * scale + increment
        if scale > 100:
            result += current
            current = 0

    return result + current
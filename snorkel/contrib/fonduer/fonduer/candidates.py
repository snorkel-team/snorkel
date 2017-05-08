from ....models.context import Document
from ....candidates import Ngrams


class OmniNgrams(Ngrams):
    """
    Defines the space of candidates as all n-grams (n <= n_max) in a Document _x_,
    divided into Phrases inside of html elements (such as table cells).
    """
    def __init__(self, n_max=5, split_tokens=['-', '/']):
        Ngrams.__init__(self, n_max=n_max, split_tokens=split_tokens)

    def apply(self, context):
        if not isinstance(context, Document):
            raise TypeError("Input Contexts to OmniNgrams.apply() must be of type Document")
        for phrase in context.phrases:
            for ts in Ngrams.apply(self, phrase):
                yield ts
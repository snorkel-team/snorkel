from ..grammar import Annotator

class TokenAnnotator(Annotator):
    def annotate(self, tokens):
        # Quotation marks are hard stops to prevent merging of multiple strings
        if len(tokens) == 1 and tokens[0]['pos'] not in ["``", "\'\'"]:
            return [('$QueryToken', tokens[0]['word'])]
        else:
            return []

annotators = [TokenAnnotator()]
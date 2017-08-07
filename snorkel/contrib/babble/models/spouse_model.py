import os

from snorkel.parser import TSVDocPreprocessor

from babble_model import BabbleModel

class SpouseModel(BabbleModel):
    def parse(self, 
              file_path=(os.environ['SNORKELHOME'] + '/tutorials/intro/data/articles.tsv'), 
              clear=True):
        doc_preprocessor = TSVDocPreprocessor(file_path, max_docs=n_docs)
        super(SpouseModel, self).parse(self, doc_preprocessor, clear=clear)

    def babble(self, explanations, user_lists={}, **kwargs):
        babbler = Babbler(self.candidate_class, explanations, user_lists=user_lists)
        super(SpouseModel, self).babble(babbler, **kwargs)
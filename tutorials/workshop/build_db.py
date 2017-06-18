import os
from snorkel import SnorkelSession
from snorkel.parser import CorpusParser
from snorkel.parser import TSVDocPreprocessor
from snorkel.models import Document, Sentence




def main(args):

    session = SnorkelSession()

    args.parser = "corenlp"



    doc_preprocessor = TSVDocPreprocessor('data/articles4.tsv', max_docs=max_docs)

    from snorkel.contrib.parser import *
    #parser = RuleBasedParser()
    #parser = Spacy()




    corpus_parser = CorpusParser()
    %time corpus_parser.apply(doc_preprocessor)



    print("Documents:", session.query(Document).count())
    print("Sentences:", session.query(Sentence).count())

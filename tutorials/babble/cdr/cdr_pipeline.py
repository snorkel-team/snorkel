import csv
import os

import numpy as np
from six.moves.cPickle import load

from snorkel.candidates import Ngrams, PretaggedCandidateExtractor
from snorkel.matchers import PersonMatcher
from snorkel.models import Document, Sentence, StableLabel
from snorkel.parser import CorpusParser, XMLMultiDocPreprocessor
from snorkel.parser.spacy_parser import Spacy

from snorkel.contrib.babble import Babbler
from snorkel.contrib.babble.pipelines import BabblePipeline
from snorkel.contrib.babble.pipelines.snorkel_pipeline import TRAIN, DEV, TEST

from tutorials.babble.cdr.utils import TaggerOneTagger
from tutorials.babble.cdr.load_external_annotations import load_external_labels
from tutorials.babble.cdr.cdr_examples import get_explanations, get_user_lists

DATA_ROOT = os.environ['SNORKELHOME'] + '/tutorials/babble/cdr/data/'

class CdrPipeline(BabblePipeline):
    def parse(self, 
              file_path=(DATA_ROOT + 'CDR.BioC.xml'),
                                    # CDR.BioC.small.xml 
              clear=True,
              config=None):
        doc_preprocessor = XMLMultiDocPreprocessor(
            path=file_path,
            doc='.//document',
            text='.//passage/text/text()',
            id='.//id/text()',
            max_docs=self.config['max_docs']
        )
        tagger_one = TaggerOneTagger()
        corpus_parser = CorpusParser(parser=Spacy(), fn=tagger_one.tag)
        corpus_parser.apply(list(doc_preprocessor), 
                    count=doc_preprocessor.max_docs, 
                    parallelism=self.config['parallelism'], 
                    clear=clear)
        if self.config['verbose']:
            print("Documents: {}".format(self.session.query(Document).count()))
            print("Sentences: {}".format(self.session.query(Sentence).count()))
        
    def extract(self, clear=True, config=None):
        with open(DATA_ROOT + 'doc_ids.pkl', 'rb') as f:
            train_ids, dev_ids, test_ids = load(f)
            train_ids, dev_ids, test_ids = set(train_ids), set(dev_ids), set(test_ids)

        train_sents, dev_sents, test_sents = set(), set(), set()
        docs = self.session.query(Document).order_by(Document.name).all()
        for i, doc in enumerate(docs):
            for s in doc.sentences:
                if doc.name in train_ids:
                    train_sents.add(s)
                elif doc.name in dev_ids:
                    dev_sents.add(s)
                elif doc.name in test_ids:
                    test_sents.add(s)
                else:
                    raise Exception('ID <{0}> not found in any id set'.format(doc.name))

        candidate_extractor = PretaggedCandidateExtractor(self.candidate_class,
                                                          ['Chemical', 'Disease'])
        
        for split, sents in enumerate([train_sents, dev_sents, test_sents]):
            if len(sents) > 0 and split in self.config['splits']:
                super(CdrPipeline, self).extract(
                    candidate_extractor, sents, split=split, clear=clear)


    def load_gold(self, config=None):
        load_external_labels(self.session, self.candidate_class, split=0, annotator='gold')
        load_external_labels(self.session, self.candidate_class, split=1, annotator='gold')
        load_external_labels(self.session, self.candidate_class, split=2, annotator='gold')

    def collect(self):
        candidates = self.get_candidates(split=self.config['babbler_candidate_split'])
        explanations = get_explanations(candidates)
        user_lists = get_user_lists()
        super(CdrPipeline, self).babble('text', explanations, user_lists, self.config)

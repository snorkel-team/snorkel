import csv
import os

import numpy as np
from six.moves.cPickle import load

from snorkel.candidates import Ngrams, PretaggedCandidateExtractor
from snorkel.matchers import PersonMatcher
from snorkel.models import Document, Sentence, StableLabel
from snorkel.parser import CorpusParser, TSVDocPreprocessor

from snorkel.contrib.babble import Babbler
from snorkel.contrib.babble.pipelines import BabblePipeline
from snorkel.contrib.babble.pipelines.snorkel_pipeline import TRAIN, DEV, TEST

from tutorials.babble.protein.utils import ProteinKinaseLookupTagger
from tutorials.babble.protein.load_external_annotations import load_external_labels
from tutorials.babble.protein.protein_examples import get_explanations, get_user_lists

DATA_ROOT = os.environ['SNORKELHOME'] + '/tutorials/babble/protein/data/'

class ProteinPipeline(BabblePipeline):
    def parse(self, 
              # file_path=(DATA_ROOT + 'abstracts_subset.txt'), 
              file_path=(DATA_ROOT + 'abstracts_razor_utf8.txt'), 
              clear=True,
              config=None):
        if 'subset' in file_path:
            print("WARNING: you are currently using a subset of the data.")
        doc_preprocessor = TSVDocPreprocessor(file_path, 
                                              max_docs=self.config['max_docs'])
        pk_lookup_tagger = ProteinKinaseLookupTagger()
        corpus_parser = CorpusParser(fn=pk_lookup_tagger.tag)
        corpus_parser.apply(list(doc_preprocessor), 
                            parallelism=self.config['parallelism'], 
                            clear=clear)
        if self.config['verbose']:
            print("Documents: {}".format(self.session.query(Document).count()))
            print("Sentences: {}".format(self.session.query(Sentence).count()))
        
    def extract(self, clear=True, config=None):
                
        with open(DATA_ROOT + 'all_pkr_ids.pkl', 'rb') as f:
        # with open(DATA_ROOT + 'subset_pkr_ids.pkl', 'rb') as f:
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
                    print("Warning: ID <{0}> not found in any id set. Adding to dev...".format(doc.name))
                    dev_sents.add(s)

        candidate_extractor = PretaggedCandidateExtractor(self.candidate_class,
                                                          ['protein', 'kinase'])
        
        for split, sents in enumerate([train_sents, dev_sents, test_sents]):
            if len(sents) > 0 and split in self.config['splits']:
                super(ProteinPipeline, self).extract(
                    candidate_extractor, sents, split=split, clear=clear)


    def load_gold(self, config=None):
        load_external_labels(self.session, self.candidate_class, split=0, annotator='gold')
        load_external_labels(self.session, self.candidate_class, split=1, annotator='gold')
        load_external_labels(self.session, self.candidate_class, split=2, annotator='gold')

    def collect(self):
        candidates = self.get_candidates(split=self.config['babbler_candidate_split'])
        explanations = get_explanations(candidates)
        user_lists = get_user_lists()
        super(ProteinPipeline, self).babble('text', explanations, user_lists, self.config)
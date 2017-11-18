import csv
import os

import numpy as np
import random
from six.moves.cPickle import load

from snorkel.annotations import load_gold_labels
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
        for split in self.config['splits']:
            load_external_labels(self.session, self.candidate_class, 
                                 split=split, annotator='gold')
        
            ### SPECIAL: Trim candidate set to meet desired positive pct
            # Process is deterministic to ensure repeatable results
            SEED = 123
            TARGET_PCT = 0.20
            positives = []
    
            candidates = self.session.query(self.candidate_class).filter(
                self.candidate_class.split == split).all()

            L_gold = load_gold_labels(self.session, annotator_name='gold', split=split)
            total = len(candidates)
            positive = float(sum(L_gold.todense() == 1))
            print("Positive % before pruning: {:.1f}%\n".format(positive/total * 100))
            
            for c in candidates:
                label = L_gold[L_gold.get_row_index(c), 0]
                if label > 0:
                    positives.append(c)
            
            target = int(TARGET_PCT * total)
            random.seed(SEED)
            to_delete = random.sample(positives, len(positives) - target)

            for c in to_delete:
                self.session.delete(c)

            L_gold = load_gold_labels(self.session, annotator_name='gold', split=split)
            positive = float(sum(L_gold.todense() == 1))
            print("Positive % after pruning: {:.1f}%\n".format(positive/total * 100))

            self.session.commit()

    def collect(self):
        candidates = self.get_candidates(split=self.config['babbler_candidate_split'])
        explanations = get_explanations()
        user_lists = get_user_lists()
        super(CdrPipeline, self).babble('text', explanations, user_lists, self.config)

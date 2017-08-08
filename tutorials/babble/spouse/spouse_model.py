import os
import numpy as np

from snorkel.candidates import Ngrams, CandidateExtractor
from snorkel.matchers import PersonMatcher
from snorkel.models import Document
from snorkel.parser import TSVDocPreprocessor
from tutorials.intro import load_external_labels, number_of_people

from snorkel.contrib.babble import Babbler
from snorkel.contrib.babble.models import BabbleModel

class SpouseModel(BabbleModel):
    def parse(self, 
              file_path=(os.environ['SNORKELHOME'] + '/tutorials/intro/data/articles.tsv'), 
              clear=True):
        doc_preprocessor = TSVDocPreprocessor(file_path, max_docs=self.config['max_docs'])
        super(SpouseModel, self).parse(doc_preprocessor, clear=clear)

    def extract(self, clear=True):
        ngrams         = Ngrams(n_max=7)
        person_matcher = PersonMatcher(longest_match_only=True)
        candidate_extractor = CandidateExtractor(
            self.candidate_class, 
            [ngrams, ngrams], 
            [person_matcher, 
            person_matcher])
        
        docs = self.session.query(Document).order_by(Document.name).all()

        train_sents = set()
        dev_sents   = set()
        test_sents  = set()

        for i, doc in enumerate(docs):
            for s in doc.sentences:
                if number_of_people(s) <= 5:
                    if i % 10 == 8:
                        dev_sents.add(s)
                    elif i % 10 == 9:
                        test_sents.add(s)
                    else:
                        train_sents.add(s)

        for split, sents in enumerate([train_sents, dev_sents, test_sents]):
            if len(sents) > 0 and split in self.config['splits']:
                super(SpouseModel, self).extract(
                    candidate_extractor, sents, split=split, clear=clear)

    def load_gold(self):
        fpath = os.environ['SNORKELHOME'] + '/tutorials/intro/data/gold_labels.tsv'
        load_external_labels(self.session, self.candidate_class, 
                             annotator_name='gold', path=fpath)

    def babble(self, explanations, user_lists={}, **kwargs):
        babbler = Babbler(mode='text', candidate_class=self.candidate_class, 
                          explanations=explanations, user_lists=user_lists)
        super(SpouseModel, self).babble(babbler, **kwargs)

    def use_intro_lfs(self):
        import re
        from snorkel.lf_helpers import (
            get_left_tokens, get_right_tokens, get_between_tokens,
            get_text_between, get_tagged_text,
        )

        spouses = {'spouse', 'wife', 'husband', 'ex-wife', 'ex-husband'}
        family = {'father', 'mother', 'sister', 'brother', 'son', 'daughter',
                    'grandfather', 'grandmother', 'uncle', 'aunt', 'cousin'}
        family = family | {f + '-in-law' for f in family}
        other = {'boyfriend', 'girlfriend' 'boss', 'employee', 'secretary', 'co-worker'}

        # Helper function to get last name
        def last_name(s):
            name_parts = s.split(' ')
            return name_parts[-1] if len(name_parts) > 1 else None    

        def LF_husband_wife(c):
            return 1 if len(spouses.intersection(get_between_tokens(c))) > 0 else 0

        def LF_husband_wife_left_window(c):
            if len(spouses.intersection(get_left_tokens(c[0], window=2))) > 0:
                return 1
            elif len(spouses.intersection(get_left_tokens(c[1], window=2))) > 0:
                return 1
            else:
                return 0
            
        def LF_same_last_name(c):
            p1_last_name = last_name(c.person1.get_span())
            p2_last_name = last_name(c.person2.get_span())
            if p1_last_name and p2_last_name and p1_last_name == p2_last_name:
                if c.person1.get_span() != c.person2.get_span():
                    return 1
            return 0

        def LF_no_spouse_in_sentence(c):
            return -1 if np.random.rand() < 0.75 and len(spouses.intersection(c.get_parent().words)) == 0 else 0

        def LF_and_married(c):
            return 1 if 'and' in get_between_tokens(c) and 'married' in get_right_tokens(c) else 0
            
        def LF_familial_relationship(c):
            return -1 if len(family.intersection(get_between_tokens(c))) > 0 else 0

        def LF_family_left_window(c):
            if len(family.intersection(get_left_tokens(c[0], window=2))) > 0:
                return -1
            elif len(family.intersection(get_left_tokens(c[1], window=2))) > 0:
                return -1
            else:
                return 0

        def LF_other_relationship(c):
            return -1 if len(other.intersection(get_between_tokens(c))) > 0 else 0

        import bz2

        # Function to remove special characters from text
        def strip_special(s):
            return ''.join(c for c in s if ord(c) < 128)

        # Read in known spouse pairs and save as set of tuples
        with bz2.BZ2File(os.environ['SNORKELHOME'] + '/tutorials/intro/data/spouses_dbpedia.csv.bz2', 'rb') as f:
            known_spouses = set(
                tuple(strip_special(x).strip().split(',')) for x in f.readlines()
            )
        # Last name pairs for known spouses
        last_names = set([(last_name(x), last_name(y)) for x, y in known_spouses if last_name(x) and last_name(y)])
            
        def LF_distant_supervision(c):
            p1, p2 = c.person1.get_span(), c.person2.get_span()
            return 1 if (p1, p2) in known_spouses or (p2, p1) in known_spouses else 0

        def LF_distant_supervision_last_names(c):
            p1, p2 = c.person1.get_span(), c.person2.get_span()
            p1n, p2n = last_name(p1), last_name(p2)
            return 1 if (p1 != p2) and ((p1n, p2n) in last_names or (p2n, p1n) in last_names) else 0

        self.lfs = [
            LF_distant_supervision, LF_distant_supervision_last_names, 
            LF_husband_wife, LF_husband_wife_left_window, LF_same_last_name,
            LF_no_spouse_in_sentence, LF_and_married, LF_familial_relationship, 
            LF_family_left_window, LF_other_relationship
        ]
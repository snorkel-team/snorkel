import os
import numpy as np

from snorkel.candidates import Ngrams, CandidateExtractor
from snorkel.matchers import PersonMatcher
from snorkel.models import Document
from snorkel.parser import TSVDocPreprocessor
from tutorials.intro import load_external_labels, number_of_people

from snorkel.contrib.babble import Babbler
from snorkel.contrib.babble.models import BabbleModel

class BikeModel(BabbleModel):

    def get_candidates(self):
        

    def load_gold(self):
        pass

    def babble(self, explanations, user_lists={}, **kwargs):
        babbler = Babbler(mode='image', candidate_class=None, explanations=explanations)
        super(BikeModel, self).babble(babbler, **kwargs)
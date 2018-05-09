from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# Hiding this import for now so that 'Spouse' in setUpClass() passes a string
# to candidate_subclass, which ultimately calls type(...). So this import can be
# re-added when Python 2 is no longer supported.
#from __future__ import unicode_literals
from builtins import *

import os
from snorkel.annotations import load_gold_labels, load_marginals
from snorkel.models import candidate_subclass
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import unittest


class PyTorchTestBase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        snorkel_engine = create_engine(os.path.join('sqlite:///' + dir_path, 'spouses.db'))
        SnorkelSession = sessionmaker(bind=snorkel_engine)
        cls.session = SnorkelSession()

        Spouse = candidate_subclass('Spouse', ['person1', 'person2'])

        cls.train_marginals = load_marginals(cls.session, split=0)

        cls.train_cands = cls.session.query(Spouse).filter(Spouse.split == 0).order_by(Spouse.id).all()
        cls.dev_cands   = cls.session.query(Spouse).filter(Spouse.split == 1).order_by(Spouse.id).all()
        cls.test_cands  = cls.session.query(Spouse).filter(Spouse.split == 2).order_by(Spouse.id).all()

        cls.L_gold_dev  = load_gold_labels(cls.session, annotator_name='gold', split=1)
        cls.L_gold_test = load_gold_labels(cls.session, annotator_name='gold', split=2)

    @classmethod
    def tearDownClass(cls):
        cls.session.close()

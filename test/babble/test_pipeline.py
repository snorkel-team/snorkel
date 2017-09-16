import unittest

from snorkel import SnorkelSession
from snorkel.models import candidate_subclass
from snorkel.contrib.babble.pipelines import config
from tutorials.babble.spouse import SpousePipeline

class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.session = SnorkelSession()

    @classmethod
    def tearDownClass(cls):
        pass

    def test_pipeline_creation(self):
        Spouse = candidate_subclass('Spouse', ['person1', 'person2'])
        pipe = SpousePipeline(self.session, Spouse, config)

suite = unittest.TestLoader().loadTestsFromTestCase(TestPipeline)
unittest.TextTestRunner(verbosity=2).run(suite)
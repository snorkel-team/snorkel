import unittest

from snorkel import SnorkelSession
from snorkel.models import candidate_subclass
from snorkel.contrib.babble.pipelines import config, STAGES

from tutorials.babble.stub.stub_pipeline import StubPipeline

class TestPipeline(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def test_pipeline_creation(self):
        print('\n')
        config['start_at'] = STAGES.COLLECT
        config['end_at'] = STAGES.SUPERVISE
        pipe = StubPipeline(None, None, config)
        pipe.run()

suite = unittest.TestLoader().loadTestsFromTestCase(TestPipeline)
unittest.TextTestRunner(verbosity=2).run(suite)
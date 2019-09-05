import json
import os
import shutil
import tempfile
import unittest

from snorkel.classification.training.loggers import TensorBoardWriter
from snorkel.types import Config


class TempConfig(Config):
    a: int = 42
    b: str = "foo"


class TestTensorBoardWriter(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_tensorboard_writer(self):
        # Note: this just tests API calls. We rely on
        # tensorboardX's unit tests for correctness.
        run_name = "my_run"
        config = TempConfig(b="bar")
        writer = TensorBoardWriter(run_name=run_name, log_dir=self.test_dir)
        writer.add_scalar("my_value", value=0.5, step=2)
        writer.write_config(config)
        log_path = os.path.join(self.test_dir, run_name, "config.json")
        with open(log_path, "r") as f:
            file_config = json.load(f)
        self.assertEqual(config._asdict(), file_config)
        writer.cleanup()

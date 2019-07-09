import os
import shutil
import tempfile
import unittest

from snorkel.classification.snorkel_classifier import SnorkelClassifier
from snorkel.classification.training import Checkpointer


class TestLogManager(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_checkpointer(self) -> None:
        checkpointer = Checkpointer(
            checkpoint_dir=self.test_dir,
            checkpoint_runway=3,
            checkpoint_metric="f1:max",
        )
        model = SnorkelClassifier([])
        checkpointer.checkpoint(2, model, dict(f1=0.5))
        self.assertEqual(len(checkpointer.best_metric_dict), 0)
        checkpointer.checkpoint(3, model, dict(f1=0.8, f2=0.5))
        self.assertEqual(checkpointer.best_metric_dict["f1"], 0.8)
        checkpointer.checkpoint(4, model, dict(f1=0.9))
        self.assertEqual(checkpointer.best_metric_dict["f1"], 0.9)

    def test_checkpointer_min(self) -> None:
        checkpointer = Checkpointer(
            checkpoint_dir=self.test_dir,
            checkpoint_runway=3,
            checkpoint_metric="f1:min",
        )
        model = SnorkelClassifier([])
        checkpointer.checkpoint(3, model, dict(f1=0.8, f2=0.5))
        self.assertEqual(checkpointer.best_metric_dict["f1"], 0.8)
        checkpointer.checkpoint(4, model, dict(f1=0.7))
        self.assertEqual(checkpointer.best_metric_dict["f1"], 0.7)

    def test_checkpointer_clear(self) -> None:
        checkpoint_dir = os.path.join(self.test_dir, "clear")
        checkpointer = Checkpointer(
            checkpoint_dir=checkpoint_dir,
            checkpoint_metric="f1:max",
            checkpoint_clear=True,
        )
        model = SnorkelClassifier([])
        checkpointer.checkpoint(1, model, dict(f1=0.8))
        expected_files = ["checkpoint_1.pth", "best_model_f1.pth"]
        self.assertEqual(set(os.listdir(checkpoint_dir)), set(expected_files))
        checkpointer.clear()
        expected_files = ["best_model_f1.pth"]
        self.assertEqual(os.listdir(checkpoint_dir), expected_files)

    def test_checkpointer_load_best(self) -> None:
        checkpoint_dir = os.path.join(self.test_dir, "clear")
        checkpointer = Checkpointer(
            checkpoint_dir=checkpoint_dir, checkpoint_metric="f1:max"
        )
        model = SnorkelClassifier([])
        checkpointer.checkpoint(1, model, dict(f1=0.8))
        load_model = checkpointer.load_best_model(model)
        self.assertEqual(model, load_model)

    def test_no_checkpoint_dir(self) -> None:
        with self.assertRaisesRegex(ValueError, "no checkpoint_dir"):
            Checkpointer(checkpoint_dir=None)

    def test_no_zero_frequency(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint freq"):
            Checkpointer(checkpoint_dir=self.test_dir, checkpoint_factor=0)

    def test_bad_metric_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "metric_name:mode"):
            Checkpointer(checkpoint_dir=self.test_dir, checkpoint_metric="f1-min")

        with self.assertRaisesRegex(ValueError, "metric mode"):
            Checkpointer(checkpoint_dir=self.test_dir, checkpoint_metric="f1:mode")

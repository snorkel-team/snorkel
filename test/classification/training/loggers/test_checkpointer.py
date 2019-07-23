import os
import shutil
import tempfile
import unittest

from snorkel.classification.snorkel_classifier import SnorkelClassifier
from snorkel.classification.training import Checkpointer

log_manager_config = {"counter_unit": "epochs", "evaluation_freq": 1}


class TestLogManager(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_checkpointer(self) -> None:
        checkpointer = Checkpointer(
            **log_manager_config,
            checkpoint_dir=self.test_dir,
            checkpoint_runway=3,
            checkpoint_metric="task/dataset/valid/f1:max",
        )
        model = SnorkelClassifier([])
        checkpointer.checkpoint(2, model, {"task/dataset/valid/f1": 0.5})
        self.assertEqual(len(checkpointer.best_metric_dict), 0)
        checkpointer.checkpoint(
            3, model, {"task/dataset/valid/f1": 0.8, "task/dataset/valid/f2": 0.5}
        )
        self.assertEqual(checkpointer.best_metric_dict["task/dataset/valid/f1"], 0.8)
        checkpointer.checkpoint(4, model, {"task/dataset/valid/f1": 0.9})
        self.assertEqual(checkpointer.best_metric_dict["task/dataset/valid/f1"], 0.9)

    def test_checkpointer_min(self) -> None:
        checkpointer = Checkpointer(
            **log_manager_config,
            checkpoint_dir=self.test_dir,
            checkpoint_runway=3,
            checkpoint_metric="task/dataset/valid/f1:min",
        )
        model = SnorkelClassifier([])
        checkpointer.checkpoint(
            3, model, {"task/dataset/valid/f1": 0.8, "task/dataset/valid/f2": 0.5}
        )
        self.assertEqual(checkpointer.best_metric_dict["task/dataset/valid/f1"], 0.8)
        checkpointer.checkpoint(4, model, {"task/dataset/valid/f1": 0.7})
        self.assertEqual(checkpointer.best_metric_dict["task/dataset/valid/f1"], 0.7)

    def test_checkpointer_clear(self) -> None:
        checkpoint_dir = os.path.join(self.test_dir, "clear")
        checkpointer = Checkpointer(
            **log_manager_config,
            checkpoint_dir=checkpoint_dir,
            checkpoint_metric="task/dataset/valid/f1:max",
            checkpoint_clear=True,
        )
        model = SnorkelClassifier([])
        checkpointer.checkpoint(1, model, {"task/dataset/valid/f1": 0.8})
        expected_files = ["checkpoint_1.pth", "best_model_task_dataset_valid_f1.pth"]
        self.assertEqual(set(os.listdir(checkpoint_dir)), set(expected_files))
        checkpointer.clear()
        expected_files = ["best_model_task_dataset_valid_f1.pth"]
        self.assertEqual(os.listdir(checkpoint_dir), expected_files)

    def test_checkpointer_load_best(self) -> None:
        checkpoint_dir = os.path.join(self.test_dir, "clear")
        checkpointer = Checkpointer(
            **log_manager_config,
            checkpoint_dir=checkpoint_dir,
            checkpoint_metric="task/dataset/valid/f1:max",
        )
        model = SnorkelClassifier([])
        checkpointer.checkpoint(1, model, {"task/dataset/valid/f1": 0.8})
        load_model = checkpointer.load_best_model(model)
        self.assertEqual(model, load_model)

    def test_no_checkpoint_dir(self) -> None:
        with self.assertRaisesRegex(ValueError, "no checkpoint_dir"):
            Checkpointer(**log_manager_config, checkpoint_dir=None)

    def test_no_zero_frequency(self) -> None:
        with self.assertRaisesRegex(ValueError, "checkpoint freq"):
            Checkpointer(
                **log_manager_config, checkpoint_dir=self.test_dir, checkpoint_factor=0
            )

    def test_bad_metric_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "metric_name:mode"):
            Checkpointer(
                **log_manager_config,
                checkpoint_dir=self.test_dir,
                checkpoint_metric="task/dataset/split/f1-min",
            )

        with self.assertRaisesRegex(ValueError, "metric mode"):
            Checkpointer(
                **log_manager_config,
                checkpoint_dir=self.test_dir,
                checkpoint_metric="task/dataset/split/f1:mode",
            )

        with self.assertRaisesRegex(ValueError, "checkpoint_metric must be formatted"):
            Checkpointer(
                **log_manager_config,
                checkpoint_dir=self.test_dir,
                checkpoint_metric="accuracy:max",
            )

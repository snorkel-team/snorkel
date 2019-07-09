import shutil
import tempfile
import unittest

from snorkel.classification.training import Checkpointer, LogManager, LogWriter
from snorkel.classification.snorkel_classifier import SnorkelClassifier


class TestLogManager(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_log_manager_points(self) -> None:
        """Unit test of log_manager (points)"""
        log_manager_config = {"counter_unit": "points", "evaluation_freq": 10}

        checkpointer = Checkpointer(
            checkpoint_dir=self.test_dir, checkpoint_factor=2, **log_manager_config
        )
        log_manager = LogManager(  # type: ignore
            n_batches_per_epoch=10, checkpointer=checkpointer, **log_manager_config
        )

        log_manager.update(5)
        self.assertFalse(log_manager.trigger_evaluation())
        self.assertFalse(log_manager.trigger_checkpointing())

        log_manager.update(5)
        self.assertTrue(log_manager.trigger_evaluation())
        self.assertFalse(log_manager.trigger_checkpointing())

        log_manager.update(10)
        self.assertTrue(log_manager.trigger_evaluation())
        self.assertTrue(log_manager.trigger_checkpointing())

        log_manager.update(5)
        self.assertFalse(log_manager.trigger_evaluation())
        self.assertFalse(log_manager.trigger_checkpointing())

        self.assertEqual(log_manager.point_count, 5)
        self.assertEqual(log_manager.point_total, 25)
        self.assertEqual(log_manager.batch_total, 4)
        self.assertEqual(log_manager.epoch_total, 0.4)

    def test_log_manager_batch(self) -> None:
        """Unit test of log_manager (batches)"""
        log_manager_config = {"counter_unit": "batches", "evaluation_freq": 2}

        checkpointer = Checkpointer(
            checkpoint_dir=self.test_dir, checkpoint_factor=2, **log_manager_config
        )
        log_manager = LogManager(  # type: ignore
            n_batches_per_epoch=5, checkpointer=checkpointer, **log_manager_config
        )

        log_manager.update(5)
        self.assertFalse(log_manager.trigger_evaluation())
        self.assertFalse(log_manager.trigger_checkpointing())

        log_manager.update(5)
        self.assertTrue(log_manager.trigger_evaluation())
        self.assertFalse(log_manager.trigger_checkpointing())

        log_manager.update(10)
        self.assertFalse(log_manager.trigger_evaluation())
        self.assertFalse(log_manager.trigger_checkpointing())

        log_manager.update(5)
        self.assertTrue(log_manager.trigger_evaluation())
        self.assertTrue(log_manager.trigger_checkpointing())

        self.assertEqual(log_manager.batch_count, 0)
        self.assertEqual(log_manager.point_total, 25)
        self.assertEqual(log_manager.batch_total, 4)
        self.assertEqual(log_manager.epoch_total, 0.8)

    def test_log_manager_epoch(self) -> None:
        """Unit test of log_manager (epochs)"""
        log_manager_config = {"counter_unit": "epochs", "evaluation_freq": 1}

        checkpointer = Checkpointer(
            checkpoint_dir=self.test_dir, checkpoint_factor=2, **log_manager_config
        )
        log_manager = LogManager(  # type: ignore
            n_batches_per_epoch=2, checkpointer=checkpointer, **log_manager_config
        )

        log_manager.update(5)
        self.assertFalse(log_manager.trigger_evaluation())
        self.assertFalse(log_manager.trigger_checkpointing())

        log_manager.update(5)
        self.assertTrue(log_manager.trigger_evaluation())
        self.assertFalse(log_manager.trigger_checkpointing())

        log_manager.update(10)
        self.assertFalse(log_manager.trigger_evaluation())
        self.assertFalse(log_manager.trigger_checkpointing())

        log_manager.update(5)
        self.assertTrue(log_manager.trigger_evaluation())
        self.assertTrue(log_manager.trigger_checkpointing())

        self.assertEqual(log_manager.batch_count, 0)
        self.assertEqual(log_manager.point_total, 25)
        self.assertEqual(log_manager.batch_total, 4)
        self.assertEqual(log_manager.epoch_total, 2)

    def test_close(self) -> None:
        log_writer = LogWriter()
        checkpointer = Checkpointer(checkpoint_dir=self.test_dir)
        log_manager = LogManager(n_batches_per_epoch=2, checkpointer=checkpointer, log_writer=log_writer)

        classifier = SnorkelClassifier([])
        best_classifier = log_manager.close(classifier)
        self.assertEqual(best_classifier, classifier)

    def test_bad_unit(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unrecognized counter_unit"):
            LogManager(n_batches_per_epoch=2, counter_unit="macaroni")

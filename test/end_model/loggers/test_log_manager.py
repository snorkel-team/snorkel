import os
import shutil
import unittest

from snorkel.end_model.loggers.checkpointer import Checkpointer
from snorkel.end_model.loggers.log_manager import LogManager

TEST_LOG_DIR = "test/mtl/loggers/logs"


class TestLogManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(TEST_LOG_DIR):
            os.makedirs(TEST_LOG_DIR)

    @classmethod
    def tearDownClass(cls):
        if os.path.exists(TEST_LOG_DIR):
            shutil.rmtree(TEST_LOG_DIR)

    def test_log_manager_points(self) -> None:
        """Unit test of log_manager (points)"""
        log_manager_config = {"counter_unit": "points", "evaluation_freq": 10}

        checkpointer = Checkpointer(
            checkpoint_dir=TEST_LOG_DIR, checkpoint_factor=2, **log_manager_config
        )
        log_manager = LogManager(
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

        assert log_manager.point_count == 5
        assert log_manager.point_total == 25
        assert log_manager.batch_total == 4
        assert log_manager.epoch_total == 0.4

    def test_log_manager_batch(self) -> None:
        """Unit test of log_manager (batches)"""
        log_manager_config = {"counter_unit": "batches", "evaluation_freq": 2}

        checkpointer = Checkpointer(
            checkpoint_dir=TEST_LOG_DIR, checkpoint_factor=2, **log_manager_config
        )
        log_manager = LogManager(
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

        assert log_manager.batch_count == 0
        assert log_manager.point_total == 25
        assert log_manager.batch_total == 4
        assert log_manager.epoch_total == 0.8

    def test_log_manager_epoch(self) -> None:
        """Unit test of log_manager (epochs)"""
        log_manager_config = {"counter_unit": "epochs", "evaluation_freq": 1}

        checkpointer = Checkpointer(
            checkpoint_dir=TEST_LOG_DIR, checkpoint_factor=2, **log_manager_config
        )
        log_manager = LogManager(
            n_batches_per_epoch=2, checkpointer=checkpointer, **log_manager_config
        )

        log_manager.update(5)
        self.assertFalse(log_manager.trigger_evaluation())
        self.assertFalse(log_manager.trigger_checkpointing())

        log_manager.update(5)
        assert log_manager.trigger_evaluation() is True
        self.assertFalse(log_manager.trigger_checkpointing())

        log_manager.update(10)
        self.assertFalse(log_manager.trigger_evaluation())
        self.assertFalse(log_manager.trigger_checkpointing())

        log_manager.update(5)
        assert log_manager.trigger_evaluation() is True
        assert log_manager.trigger_checkpointing() is True

        assert log_manager.epoch_count == 0
        assert log_manager.point_total == 25
        assert log_manager.batch_total == 4
        assert log_manager.epoch_total == 2

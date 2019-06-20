import os
import shutil
import unittest

from snorkel.mtl.loggers.log_manager import LogManager

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
        config = {
            "log_dir": TEST_LOG_DIR,
            "counter_unit": "points",
            "evaluation_freq": 10,
            "checkpointing": True,
            "checkpointer_config": {"checkpoint_factor": 2},
        }

        log_manager = LogManager(n_batches_per_epoch=10, **config)

        log_manager.update(5)
        assert log_manager.trigger_evaluation() is False
        assert log_manager.trigger_checkpointing() is False

        log_manager.update(5)
        assert log_manager.trigger_evaluation() is True
        assert log_manager.trigger_checkpointing() is False

        log_manager.update(10)
        assert log_manager.trigger_evaluation() is True
        assert log_manager.trigger_checkpointing() is True

        log_manager.update(5)
        assert log_manager.trigger_evaluation() is False
        assert log_manager.trigger_checkpointing() is False

        assert log_manager.point_count == 5
        assert log_manager.point_total == 25
        assert log_manager.batch_total == 4
        assert log_manager.epoch_total == 0.4

    def test_log_manager_batch(self) -> None:
        """Unit test of log_manager (batches)"""

        config = {
            "counter_unit": "batches",
            "evaluation_freq": 2,
            "checkpointing": True,
            "checkpointer_config": {"checkpoint_factor": 2},
        }

        log_manager = LogManager(n_batches_per_epoch=5, **config)

        log_manager.update(5)
        assert log_manager.trigger_evaluation() is False
        assert log_manager.trigger_checkpointing() is False

        log_manager.update(5)
        assert log_manager.trigger_evaluation() is True
        assert log_manager.trigger_checkpointing() is False

        log_manager.update(10)
        assert log_manager.trigger_evaluation() is False
        assert log_manager.trigger_checkpointing() is False

        log_manager.update(5)
        assert log_manager.trigger_evaluation() is True
        assert log_manager.trigger_checkpointing() is True

        assert log_manager.batch_count == 0
        assert log_manager.point_total == 25
        assert log_manager.batch_total == 4
        assert log_manager.epoch_total == 0.8

    def test_log_manager_epoch(self) -> None:
        """Unit test of log_manager (epochs)"""
        config = {
            "counter_unit": "epochs",
            "evaluation_freq": 1,
            "checkpointing": True,
            "checkpointer_config": {"checkpoint_factor": 2},
        }

        log_manager = LogManager(n_batches_per_epoch=2, **config)

        log_manager.update(5)
        assert log_manager.trigger_evaluation() is False
        assert log_manager.trigger_checkpointing() is False

        log_manager.update(5)
        assert log_manager.trigger_evaluation() is True
        assert log_manager.trigger_checkpointing() is False

        log_manager.update(10)
        assert log_manager.trigger_evaluation() is False
        assert log_manager.trigger_checkpointing() is False

        log_manager.update(5)
        assert log_manager.trigger_evaluation() is True
        assert log_manager.trigger_checkpointing() is True

        assert log_manager.epoch_count == 0
        assert log_manager.point_total == 25
        assert log_manager.batch_total == 4
        assert log_manager.epoch_total == 2

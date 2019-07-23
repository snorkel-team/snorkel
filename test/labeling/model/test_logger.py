import unittest

from snorkel.labeling.model.logger import Logger


class LoggerTest(unittest.TestCase):
    def test_basic(self):
        metrics_dict = {"train/loss": 0.01}
        logger = Logger(log_freq=1)
        logger.log(metrics_dict)

        metrics_dict = {"train/message": "well done!"}
        logger = Logger(log_freq=1)
        logger.log(metrics_dict)

    def test_bad_metrics_dict(self):
        bad_metrics_dict = {"task1/slice1/train/loss": 0.05}
        logger = Logger(log_freq=1)
        self.assertRaises(Exception, logger.log, bad_metrics_dict)

    def test_valid_metrics_dict(self):
        mtl_metrics_dict = {"task1/valid/loss": 0.05}
        logger = Logger(log_freq=1)
        logger.log(mtl_metrics_dict)


if __name__ == "__main__":
    unittest.main()

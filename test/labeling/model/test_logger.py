import unittest

from snorkel.labeling.model.logger import Logger


class LoggerTest(unittest.TestCase):
    def test_bad_metrics_dict(self):
        mtl_metrics_dict = {"task1/train/loss": 0.05}
        bad_metrics_dict = {"task1/slice1/train/loss": 0.05}

        logger = Logger(log_train_every=1)
        logger.print_to_screen(mtl_metrics_dict)
        self.assertRaises(Exception, logger.print_to_screen, bad_metrics_dict)


if __name__ == "__main__":
    unittest.main()

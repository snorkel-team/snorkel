import json
import os
import shutil
import tempfile
import unittest

from snorkel.classification.training import LogWriter


class TestLogWriter(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_log_writer(self):
        run_name = "my_run"
        log_writer = LogWriter(run_name=run_name, log_root=self.test_dir)
        log_writer.add_scalar("my_value", value=0.5, step=2)

        log_filename = "my_log.json"
        log_writer.write_log(log_filename)

        log_path = os.path.join(self.test_dir, run_name, log_filename)
        with open(log_path, "r") as f:
            log = json.load(f)

        log_expected = dict(my_value=[[2, 0.5]])
        self.assertEqual(log, log_expected)

    def test_write_text(self) -> None:
        run_name = "my_run"
        filename = "my_text.txt"
        text = "my log text"
        log_writer = LogWriter(run_name=run_name, log_root=self.test_dir)
        log_writer.write_text(text, filename)
        log_path = os.path.join(self.test_dir, run_name, filename)
        with open(log_path, "r") as f:
            file_text = f.read()
        self.assertEqual(text, file_text)

    def test_write_config(self) -> None:
        run_name = "my_run"
        config = dict(a=8, b="my text")
        log_writer = LogWriter(run_name=run_name, log_root=self.test_dir)
        log_writer.write_config(config)
        log_path = os.path.join(self.test_dir, run_name, "config.json")
        with open(log_path, "r") as f:
            file_config = json.load(f)
        self.assertEqual(config, file_config)

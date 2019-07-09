import json
import logging
import os
from collections import defaultdict
from datetime import datetime

from snorkel.classification.snorkel_config import default_config
from snorkel.classification.utils import recursive_merge_dicts


class LogWriter:
    """A class for writing logs"""

    def __init__(self, **kwargs):
        self.config = recursive_merge_dicts(default_config["log_writer_config"], kwargs)

        date = datetime.now().strftime("%Y_%m_%d")
        time = datetime.now().strftime("%H_%M_%S")
        self.run_name = self.config["run_name"] or f"{date}/{time}/"

        self.log_dir = os.path.join(self.config["log_dir"], self.run_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.run_log = defaultdict(list)

    def add_scalar(self, name, value, step):
        """Log a scalar variable"""
        self.run_log[name].append((step, value))

    def write_config(self, config, config_filename="config.json"):
        """Dump the config to file

        This gets a special method for children classes that add functionality
        """
        self.write_json(config, config_filename)

    def write_log(self, log_filename):
        """Dump the run log to file"""
        self.write_json(self.run_log, log_filename)

    def write_text(self, text, filename):
        """Dump user-provided text to filename (e.g., the launch command)"""
        text_path = os.path.join(self.log_dir, filename)
        with open(text_path, "w") as f:
            f.write(text)

    def write_json(self, dict_to_write, filename):
        """Dump the log to file"""
        if not filename.endswith(".json"):
            logging.warning(
                f"Using write_json() method with a filename without a .json extension: {filename}"
            )
        log_path = os.path.join(self.log_dir, filename)
        with open(log_path, "w") as f:
            json.dump(dict_to_write, f)

    def close(self):
        pass

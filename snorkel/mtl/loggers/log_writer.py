import json
import os
from collections import defaultdict


class LogWriter:
    """A class for logging during training process.

    :param object: [description]
    :type object: [type]
    """

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.run_log = defaultdict(list)

    def add_scalar(self, name, value, step):
        """Log a scalar variable"""
        self.run_log[name].append((step, value))

    def write_config(self, config, config_filename="config.json"):
        """Dump the config to file"""
        config_path = os.path.join(self.log_dir, config_filename)
        with open(config_path, "w") as f:
            json.dump(config, f)

    def write_log(self, log_filename="log.json"):
        """Dump the log to file"""
        log_path = os.path.join(self.log_dir, log_filename)
        with open(log_path, "w") as f:
            json.dump(self.run_log, f)

    def close(self):
        pass

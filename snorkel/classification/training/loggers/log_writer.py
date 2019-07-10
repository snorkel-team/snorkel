import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, DefaultDict, List, Mapping

from snorkel.classification.snorkel_config import default_config
from snorkel.classification.utils import recursive_merge_dicts
from snorkel.types import Config


class LogWriter:
    """A class for writing logs"""

    def __init__(self, **kwargs: Any) -> None:
        assert isinstance(default_config["log_writer_config"], dict)
        self.config = recursive_merge_dicts(default_config["log_writer_config"], kwargs)

        date = datetime.now().strftime("%Y_%m_%d")
        time = datetime.now().strftime("%H_%M_%S")
        self.run_name = self.config["run_name"] or f"{date}/{time}/"

        self.log_dir = os.path.join(self.config["log_dir"], self.run_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.run_log: DefaultDict[str, List[List[float]]] = defaultdict(list)

    def add_scalar(self, name: str, value: float, step: float) -> None:
        """Log a scalar variable"""
        # Note: storing as list for JSON roundtripping
        self.run_log[name].append([step, value])

    def write_config(
        self, config: Config, config_filename: str = "config.json"
    ) -> None:
        """Dump the config to file

        This gets a special method for children classes that add functionality
        """
        self.write_json(config, config_filename)

    def write_log(self, log_filename: str) -> None:
        """Dump the run log to file"""
        self.write_json(self.run_log, log_filename)

    def write_text(self, text: str, filename: str) -> None:
        """Dump user-provided text to filename (e.g., the launch command)"""
        text_path = os.path.join(self.log_dir, filename)
        with open(text_path, "w") as f:
            f.write(text)

    def write_json(self, dict_to_write: Mapping[str, Any], filename: str) -> None:
        """Dump the log to file"""
        if not filename.endswith(".json"):  # pragma: no cover
            logging.warning(
                f"Using write_json() method with a filename without a .json extension: {filename}"
            )
        log_path = os.path.join(self.log_dir, filename)
        with open(log_path, "w") as f:
            json.dump(dict_to_write, f)

    def close(self) -> None:
        pass

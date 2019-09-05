import json
import logging
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, DefaultDict, List, Mapping, Optional

from snorkel.types import Config


class LogWriterConfig(Config):
    """Manager for checkpointing model.

    Parameters
    ----------
    log_dir
        The root directory where logs should be saved
    run_name
        The name of this particular run (defaults to date-time combination if None)
    """

    log_dir: str = "logs"
    run_name: Optional[str] = None


class LogWriter:
    """A class for writing logs.

    Parameters
    ----------
    kwargs
        Settings to merge into LogWriterConfig

    Attributes
    ----------
    config
        Merged configuration
    run_name
        Name of run if provided, otherwise date-time combination
    log_dir
        The root directory where logs should be saved
    run_log
        Dictionary of scalar values to log, keyed by value name
    """

    def __init__(self, **kwargs: Any) -> None:
        self.config = LogWriterConfig(**kwargs)

        self.run_name = self.config.run_name
        if self.run_name is None:
            date = datetime.now().strftime("%Y_%m_%d")
            time = datetime.now().strftime("%H_%M_%S")
            self.run_name = f"{date}/{time}/"

        self.log_dir = os.path.join(self.config.log_dir, self.run_name)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.run_log: DefaultDict[str, List[List[float]]] = defaultdict(list)

    def add_scalar(self, name: str, value: float, step: float) -> None:
        """Log a scalar variable.

        Parameters
        ----------
        name
            Name of the scalar collection
        value
            Value of scalar
        step
            Step axis value
        """
        # Note: storing as list for JSON roundtripping
        self.run_log[name].append([step, value])

    def write_config(
        self, config: Config, config_filename: str = "config.json"
    ) -> None:
        """Dump the config to file.

        Parameters
        ----------
        config
            JSON-compatible config to write to file
        config_filename
            Name of file in logging directory to write to
        """
        self.write_json(config._asdict(), config_filename)

    def write_log(self, log_filename: str) -> None:
        """Dump the scalar value log to file.

        Parameters
        ----------
        log_filename
            Name of file in logging directory to write to
        """
        self.write_json(self.run_log, log_filename)

    def write_text(self, text: str, filename: str) -> None:
        """Dump user-provided text to filename (e.g., the launch command).

        Parameters
        ----------
        text
            Text to write
        filename
            Name of file in logging directory to write to
        """
        text_path = os.path.join(self.log_dir, filename)
        with open(text_path, "w") as f:
            f.write(text)

    def write_json(self, dict_to_write: Mapping[str, Any], filename: str) -> None:
        """Dump a JSON-compatbile object to root log directory.

        Parameters
        ----------
        dict_to_write
            JSON-compatbile object to log
        filename
            Name of file in logging directory to write to
        """
        if not filename.endswith(".json"):  # pragma: no cover
            logging.warning(
                f"Using write_json() method with a filename without a .json extension: {filename}"
            )
        log_path = os.path.join(self.log_dir, filename)
        with open(log_path, "w") as f:
            json.dump(dict_to_write, f)

    def cleanup(self) -> None:
        """Perform final operations and close writer if necessary."""
        self.write_log("log.json")

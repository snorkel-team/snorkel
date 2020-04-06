from typing import Any

from torch.utils.tensorboard import SummaryWriter

from snorkel.types import Config

from .log_writer import LogWriter


class TensorBoardWriter(LogWriter):
    """A class for logging to Tensorboard during training process.

    See ``LogWriter`` for more attributes.

    Parameters
    ----------
    kwargs
        Passed to ``LogWriter`` initializer

    Attributes
    ----------
    writer
        ``SummaryWriter`` for logging and visualization
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.writer = SummaryWriter(self.log_dir)

    def add_scalar(self, name: str, value: float, step: float) -> None:
        """Log a scalar variable to TensorBoard.

        Parameters
        ----------
        name
            Name of the scalar collection
        value
            Value of scalar
        step
            Step axis value
        """
        self.writer.add_scalar(name, value, step)

    def write_config(
        self, config: Config, config_filename: str = "config.json"
    ) -> None:
        """Dump the config to file and add it to TensorBoard.

        Parameters
        ----------
        config
            JSON-compatible config to write to TensorBoard
        config_filename
            File to write config to
        """
        super().write_config(config, config_filename)
        self.writer.add_text(tag="config", text_string=str(config))

    def cleanup(self) -> None:
        """Close the ``SummaryWriter``."""
        self.writer.close()

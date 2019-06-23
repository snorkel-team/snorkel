from tensorboardX import SummaryWriter

from snorkel.types import Config

from .log_writer import LogWriter


class TensorBoardWriter(LogWriter):
    """A class for logging to Tensorboard during training process."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = SummaryWriter(self.log_dir)

    def add_scalar(self, name, value, step):
        """Log a scalar variable"""
        self.writer.add_scalar(name, value, step)

    def write_config(self, config: Config, config_filename: str = "config.json"):
        """Dump the config to file and add it to TensorBoard"""
        super().write_config(config, config_filename)
        self.writer.add_text(tag="config", text_string=str(config))

    def close(self):
        self.writer.close()

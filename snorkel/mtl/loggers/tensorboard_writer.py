from tensorboardX import SummaryWriter

from snorkel.types import Config

from .log_writer import LogWriter


class TensorBoardWriter(LogWriter):
    """A class for logging to Tensorboard during training process."""

    def __init__(self, log_dir):
        super().__init__(log_dir)
        self.writer = SummaryWriter(log_dir)

    def add_scalar(self, name, value, step):
        """Log a scalar variable"""
        self.writer.add_scalar(name, value, step)

    def write_config(self, config: Config, config_filename: str = "config.json"):
        """Dump the config to file"""
        super().write_config(config, config_filename)
        self.writer.add_text(tag="config", text_string=config)

    def close(self):
        self.writer.close()

import json

from tensorboardX import SummaryWriter

from .writer import LogWriter


class TensorBoardWriter(LogWriter):
    """Class for logging to Tensorboard during runs, as well as writing simple
    JSON logs at end of runs.

    Stores logs in log_dir/{YYYY}_{MM}_{DD}/{H}_{M}_{S}_run_name.json by default.
    """

    def __init__(self, log_dir=None, run_dir=None, run_name=None, **kwargs):
        super().__init__(log_dir=log_dir, run_dir=run_dir, run_name=run_name, **kwargs)

        # Set up TensorBoard summary writer
        self.tb_writer = SummaryWriter(self.log_subdir, filename_suffix=f".{run_name}")

    def add_scalar(self, name, val, i):
        if super().add_scalar(name, val, i):
            self.tb_writer.add_scalar(name, val, i)

    def write_config(self, config, *args, **kwargs):
        config_txt = json.dumps(self._sanitize_config(config), indent=1)
        self.tb_writer.add_text(tag="config", text_string=config_txt, global_step=0)
        super().write_config(config, *args, **kwargs)

    def close(self):
        self.tb_writer.close()

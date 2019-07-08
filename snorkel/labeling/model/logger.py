from collections import defaultdict

from snorkel.analysis.metrics import METRICS as standard_metric_names, metric_score


class Logger:
    """Tracks when it is time to calculate train/valid metrics and logs them"""

    def __init__(self, config, writer={}, epoch_size=None, verbose=True):
        # Strip split name from config keys
        self.config = config
        self.writer = writer
        self.verbose = verbose
        self.log_unit = self.config["log_unit"]
        self.epoch_size = epoch_size
        self.example_count = 0
        self.example_total = 0
        self.unit_count = 0
        self.unit_total = 0
        self.log_count = 0  # Count how many times logging has occurred

        # Normalize all target metric names to include split prefix
        self.log_train_metrics = [
            self.add_split_prefix(m, "train") for m in self.config["log_train_metrics"]
        ]
        self.log_valid_metrics = [
            self.add_split_prefix(m, "valid") for m in self.config["log_valid_metrics"]
        ]

        # Calculate how many log_train steps to take per log_valid steps
        self.valid_every_X = self._calculate_valid_frequency()

    def check(self, batch_size):
        """Returns True if the logging frequency has been met."""
        self.increment(batch_size)
        return self.unit_count >= self.config["log_train_every"]

    def increment(self, batch_size):
        """Update the total and relative unit counts"""
        self.example_count += batch_size
        self.example_total += batch_size
        if self.log_unit == "examples":
            self.unit_count = self.example_count
            self.unit_total = self.example_total
        elif self.log_unit == "batches":
            self.unit_count += 1
            self.unit_total += 1
        elif self.log_unit == "epochs":
            # Track epoch by example count because otherwise we only know when
            # a new epoch starts, not when an epoch ends
            if self.example_count >= self.epoch_size:
                self.unit_count += 1
                self.unit_total += 1
        else:
            raise Exception(f"Unrecognized log_unit: {self.log_unit}")

    def calculate_metrics(self, model, train_loader, valid_loader, metrics_dict):
        """Add standard and custom metrics to metrics_dict"""
        # Check whether or not it's time for validation as well
        self.log_count += 1
        log_valid = (
            valid_loader is not None
            and self.valid_every_X
            and not (self.log_count % self.valid_every_X)
        )

        metrics_dict = {}

        # Calculate custom metrics
        if self.config["log_train_metrics_func"] is not None:
            func = self.config["log_train_metrics_func"]
            func_list = func if isinstance(func, list) else [func]
            for func in func_list:
                metrics_dict = self._calculate_custom_metrics(
                    model, train_loader, func, metrics_dict, split="train"
                )
        if self.config["log_valid_metrics_func"] is not None and log_valid:
            func = self.config["log_valid_metrics_func"]
            func_list = func if isinstance(func, list) else [func]
            for func in func_list:
                metrics_dict = self._calculate_custom_metrics(
                    model, valid_loader, func, metrics_dict, split="valid"
                )

        # Calculate standard metrics
        metrics_dict = self._calculate_standard_metrics(
            model, train_loader, self.log_train_metrics, metrics_dict, "train"
        )

        if log_valid:
            metrics_dict = self._calculate_standard_metrics(
                model, valid_loader, self.log_valid_metrics, metrics_dict, "valid"
            )

        return metrics_dict

    def _calculate_custom_metrics(self, model, data_loader, func, metrics_dict, split):
        custom_metrics = func(model, data_loader)
        # Normalize all custom metrics to include split prefix
        for metric, value in custom_metrics.items():
            metric = self.add_split_prefix(metric, split)
            metrics_dict[metric] = value
        return metrics_dict

    def _calculate_standard_metrics(
        self, model, data_loader, target_metrics, metrics_dict, split
    ):
        target_standard_metrics = []
        for split_metric in target_metrics:
            metric = self.remove_split_prefix(split_metric)
            if metric in standard_metric_names:
                target_standard_metrics.append(metric)

        # Only calculate predictions if at least one standard metric requires it
        if target_standard_metrics:
            if model.multitask:
                # For multitask models, use score method for aggregation
                # This may cause inefficiency if there are multiple desired metrics
                # and we re-predict for each one.
                for metric in target_standard_metrics:
                    score = model.score(data_loader, metric, verbose=False)
                    metrics_dict[self.add_split_prefix(metric, split)] = score
            else:
                # For singletask models, predict once and use Y_probs/Y_preds
                # for all metrics calculations
                Y_preds, Y, Y_probs = model._get_predictions(
                    data_loader, return_probs=True
                )
                for metric in target_standard_metrics:
                    score = metric_score(
                        golds=Y, preds=Y_preds, probs=Y_probs, metric=metric
                    )
                    metrics_dict[self.add_split_prefix(metric, split)] = score
        return metrics_dict

    @staticmethod
    def add_split_prefix(metric, split):
        """Add split name to metric name if it is not already present

        The order of metric name components should either be:
        - task/split/metric in the multitask setting (expand to this from task/metric)
        - split/metric in the singletask setting (expand to this from metric)
        """
        if f"{split}/" in metric:
            full_metric = metric
        else:
            if "/" in metric:
                # It has two parts but not split, so must be task/metric
                task, metric = metric.split("/")
                full_metric = f"{task}/{split}/{metric}"
            else:
                # It has one part but not split, so must be metric
                full_metric = f"{split}/{metric}"
        return full_metric

    @staticmethod
    def remove_split_prefix(metric):
        """Remove prefixes from begininng of metric name (e.g., task/split/metric)"""
        return metric.split("/")[-1]

    def _calculate_valid_frequency(self):
        assert isinstance(self.config["log_train_every"], int)
        if self.config["log_valid_every"]:
            assert isinstance(self.config["log_valid_every"], int)
            if (
                self.config["log_valid_every"] < self.config["log_train_every"]
                or self.config["log_valid_every"] % self.config["log_train_every"]
            ):
                raise Exception(
                    f"Parameter `log_valid_every` ({self.config['log_valid_every']}) "
                    f"must be a multiple of `log_train_every` "
                    f"({self.config['log_train_every']})."
                )
            return int(self.config["log_valid_every"] / self.config["log_train_every"])
        else:
            return 0

    def log(self, metrics_dict):
        """Print calculated metrics and optionally write to file (json/tb)"""
        if self.writer:
            self.write_to_file(metrics_dict)

        if self.verbose:
            self.print_to_screen(metrics_dict)
        self.reset()

    def print_to_screen(self, metrics_dict):
        """Print all metrics in metrics_dict to screen"""
        score_strings = defaultdict(list)
        for full_name, value in metrics_dict.items():
            if full_name.count("/") == 2:
                task, split, metric = full_name.split("/")
            elif full_name.count("/") == 1:
                task = None
                split, metric = full_name.split("/")
            else:
                msg = f"Metric should have form task/split/metric or split/metric, not: {full_name}"
                raise Exception(msg)

            if task:
                metric_name = f"{task}/{metric}"
            else:
                metric_name = metric
            if isinstance(value, float):
                score_strings[split].append(f"{metric_name}={value:0.3f}")
            else:
                score_strings[split].append(f"{metric_name}={value}")

        header = f"{self.unit_total} {self.log_unit[:3]}"
        if self.log_unit != "epochs":
            epochs = self.example_total / self.epoch_size
            header += f" ({epochs:0.2f} epo)"
        string = f"[{header}]:"

        if score_strings["train"]:
            train_scores = f"{', '.join(score_strings['train'])}"
            string += f" TRAIN:[{train_scores}]"
        if score_strings["valid"]:
            valid_scores = f"{', '.join(score_strings['valid'])}"
            string += f" VALID:[{valid_scores}]"
        print(string)

    def write_to_file(self, metrics_dict):
        for metric, value in metrics_dict.items():
            self.writer.add_scalar(metric, value, self.unit_total)

    def reset(self):
        self.unit_count = 0
        self.example_count = 0

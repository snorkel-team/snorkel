from collections import defaultdict


class Logger:
    def __init__(self, log_train_every):
        self.log_train_every = log_train_every
        self.unit_count = 0

    def check(self):
        """Returns True if the logging frequency has been met."""
        self.unit_count += 1
        return self.unit_count >= self.log_train_every

    def log(self, metrics_dict):
        """Print calculated metrics and optionally write to file (json/tb)"""
        self.print_to_screen(metrics_dict)
        self.unit_count = 0

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

        string = f"[{self.unit_count} epochs]:"

        if score_strings["train"]:
            train_scores = f"{', '.join(score_strings['train'])}"
            string += f" TRAIN:[{train_scores}]"
        if score_strings["valid"]:
            valid_scores = f"{', '.join(score_strings['valid'])}"
            string += f" VALID:[{valid_scores}]"
        print(string)

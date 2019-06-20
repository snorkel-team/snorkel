from snorkel.model.utils import recursive_merge_dicts

from .end_model import EndModel


class LogisticRegression(EndModel):
    """A logistic regression classifier for a single-task problem"""

    def __init__(self, input_dim, output_dim=2, **kwargs):
        layer_out_dims = [input_dim, output_dim]
        overrides = {"input_batchnorm": False, "input_dropout": 0.0}
        kwargs = recursive_merge_dicts(
            kwargs, overrides, misses="insert", verbose=False
        )
        super().__init__(layer_out_dims, **kwargs)

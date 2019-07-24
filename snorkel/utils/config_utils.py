from typing import Any, Dict

from snorkel.types import Config


def merge_config(config: Config, config_updates: Dict[str, Any]) -> Config:
    """Merge a (potentially nested) dict of kwargs into a config (NamedTuple).

    Parameters
    ----------
    config
        An instantiated Config to update
    config_updates
        A potentially nested dict of settings to update in the Config

    Returns
    -------
    Config
        The updated Config

    Example
    -------
    ```
    config_updates = {
        "n_epochs": 5,
        "optimizer_config": {
            "lr": 0.001,
        }
    }
    trainer_config = merge_config(TrainerConfig(), config_updates)
    ```
    """
    for key, value in config_updates.items():
        if isinstance(value, dict):
            config_updates[key] = merge_config(getattr(config, key), value)
    return config._replace(**config_updates)

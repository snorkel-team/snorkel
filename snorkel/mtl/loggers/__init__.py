logging_default_config = {
    "counter_unit": "batches",  # [points, batches, epochs]
    "log_dir": "logs",  # The path to the directory under which logs will be written
    "evaluation_freq": 2,  # Evaluate performance every this many counter_units
    "writer_config": {"writer": "tensorboard", "verbose": True},  # [json, tensorboard]
    "checkpointing": True,
    "checkpointer_config": {
        "checkpoint_dir": None,
        "checkpoint_factor": 1,  # Checkpoint every this many evaluations
        "checkpoint_metric": "model/train/all/loss:min",
        "checkpoint_task_metrics": None,  # task_metric_name:mode
        "checkpoint_runway": 0,  # checkpointing runway (no checkpointing before k unit)
        "checkpoint_clear": True,  # whether to clear intermediate checkpoints
    },
}

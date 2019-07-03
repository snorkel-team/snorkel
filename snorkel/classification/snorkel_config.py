default_config = {
    "seed": None,  # Random seed for reproducibility; if None, seed is not set.
    "n_epochs": 1,  # total number of learning epochs
    "train_split": "train",  # the split for training, accepts str or list of strs
    "valid_split": "valid",  # the split for validation, accepts str or list of strs
    "test_split": "test",  # the split for testing, accepts str or list of strs
    "progress_bar": True,
    "model_config": {
        "model_path": None,  # the path to a saved checkpoint to initialize with
        "device": 0,  # gpu id (int) or -1 for cpu
        "dataparallel": True,
    },
    "log_manager_config": {
        "counter_unit": "epochs",  # [points, batches, epochs]
        "evaluation_freq": 1.0,  # Evaluate performance every this many counter_units
    },
    "checkpointing": False,  # Whether to save checkpoints of best performing models
    "checkpointer_config": {  # Note that checkpointer behavior also depends on log_manager_config
        "checkpoint_dir": None,  # Trainer will set this to log_dir if None
        "checkpoint_factor": 1,  # Checkpoint every this many evaluations
        "checkpoint_metric": "model/all/train/loss:min",
        "checkpoint_task_metrics": None,  # task_metric_name:mode
        "checkpoint_runway": 0,  # checkpointing runway (no checkpointing before k unit)
        "checkpoint_clear": True,  # whether to clear intermediate checkpoints
    },
    "logging": False,  # Whether to write logs (to json/tensorboard)
    "log_writer_config": {
        "writer": "tensorboard",  # [json, tensorboard]
        "log_root": "logs",  # The path to the root of the directory where logs are written
        "run_name": None,  # The name of this particular run (default to date/time)
    },
    "optimizer_config": {
        "optimizer": "adam",  # [sgd, adam]
        "lr": 0.001,  # learing rate
        "l2": 0.0,  # l2 regularization
        "grad_clip": 1.0,  # gradient clipping
        "sgd_config": {"momentum": 0.9},
        "adam_config": {"betas": (0.9, 0.999), "amsgrad": False},
        "adamax_config": {"betas": (0.9, 0.999), "eps": 0.00000001},
    },
    "lr_scheduler_config": {
        "lr_scheduler": "constant",  # [constant, linear, exponential, step, multi_step, reduce_on_plateau]
        "warmup_steps": 0,  # warm up steps
        "warmup_unit": "batches",  # [epochs, batches]
        "warmup_percentage": 0.0,  # warm up percentage
        "min_lr": 0.0,  # minimum learning rate
        "exponential_config": {"gamma": 0.9},
        "plateau_config": {"factor": 0.5, "patience": 10, "threshold": 0.0001},
    },
    "batch_scheduler": "shuffled",  # [sequential, shuffled]
}

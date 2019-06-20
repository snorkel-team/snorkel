em_default_config = {
    # GENERAL
    "seed": None,
    "verbose": True,
    "show_plots": True,
    # Network
    # The first value is the output dim of the input module (or the sum of
    # the output dims of all the input modules if multitask=True and
    # multiple input modules are provided). The last value is the
    # output dim of the head layer (i.e., the cardinality of the
    # classification task). The remaining values are the output dims of
    # middle layers (if any). The number of middle layers will be inferred
    # from this list.
    "layer_out_dims": [10, 2],
    # Input layer configs
    "input_layer_config": {
        "input_relu": True,
        "input_batchnorm": False,
        "input_dropout": 0.0,
    },
    # Middle layer configs
    "middle_layer_config": {
        "middle_relu": True,
        "middle_batchnorm": False,
        "middle_dropout": 0.0,
    },
    # Can optionally skip the head layer completely, for e.g. running baseline
    # models...
    "skip_head": False,
    # Device
    "device": "cpu",
    # TRAINING
    "train_config": {
        # Loss function config
        "loss_fn_reduction": "mean",
        # Display
        "progress_bar": False,
        # Dataloader
        "data_loader_config": {"batch_size": 32, "num_workers": 1, "shuffle": True},
        # Loss weights
        "loss_weights": None,
        # Train Loop
        "n_epochs": 10,
        # 'grad_clip': 0.0,
        "l2": 0.0,
        "validation_metric": "accuracy",
        "validation_freq": 1,
        "validation_scoring_kwargs": {},
        # Evaluate dev for during training every this many epochs
        # Optimizer
        "optimizer_config": {
            "optimizer": "adam",
            "optimizer_common": {"lr": 0.01},
            # Optimizer - SGD
            "sgd_config": {"momentum": 0.9},
            # Optimizer - Adam
            "adam_config": {"betas": (0.9, 0.999)},
            # Optimizer - RMSProp
            "rmsprop_config": {},  # Use defaults
        },
        # LR Scheduler (for learning rate)
        "lr_scheduler": "reduce_on_plateau",
        # [None, 'exponential', 'reduce_on_plateau']
        # 'reduce_on_plateau' uses checkpoint_metric to assess plateaus
        "lr_scheduler_config": {
            # Freeze learning rate initially this many epochs
            "lr_freeze": 0,
            # Scheduler - exponential
            "exponential_config": {"gamma": 0.9},  # decay rate
            # Scheduler - reduce_on_plateau
            "plateau_config": {
                "factor": 0.5,
                "patience": 10,
                "threshold": 0.0001,
                "min_lr": 1e-4,
            },
        },
        # Logger (see metal/logging/logger.py for descriptions)
        "logger": True,
        "logger_config": {
            "log_unit": "epochs",  # ['seconds', 'examples', 'batches', 'epochs']
            "log_train_every": 1,  # How often train metrics are calculated (optionally logged to TB)
            "log_train_metrics": [
                "loss"
            ],  # Metrics to calculate and report every `log_train_every` units. This can include built-in and user-defined metrics.
            "log_train_metrics_func": None,  # A function or list of functions that map a model + train_loader to a dictionary of custom metrics
            "log_valid_every": 1,  # How frequently to evaluate on valid set (must be multiple of log_freq)
            "log_valid_metrics": [
                "accuracy"
            ],  # Metrics to calculate and report every `log_valid_every` units; this can include built-in and user-defined metrics
            "log_valid_metrics_func": None,  # A function or list of functions that maps a model + valid_loader to a dictionary of custom metrics
        },
        # LogWriter/Tensorboard (see metal/logging/writer.py for descriptions)
        "writer": None,  # [None, "json", "tensorboard"]
        "writer_config": {  # Log (or event) file stored at log_dir/run_dir/run_name
            "log_dir": None,
            "run_dir": None,
            "run_name": None,
            "writer_metrics": None,  # May specify a subset of metrics in metrics_dict to be written
            "include_config": True,  # If True, include model config in log
        },
        # Checkpointer (see metal/logging/checkpointer.py for descriptions)
        "checkpoint": True,  # If True, checkpoint models when certain conditions are met
        "checkpoint_config": {
            "checkpoint_best": True,
            "checkpoint_every": None,  # uses log_valid_unit for units; if not None, checkpoint this often regardless of performance
            "checkpoint_metric": "accuracy",  # Must be in metrics dict; assumes valid split unless appended with "train/"
            "checkpoint_metric_mode": "max",  # ['max', 'min']
            "checkpoint_dir": "checkpoints",
            "checkpoint_runway": 0,
        },
    },
}

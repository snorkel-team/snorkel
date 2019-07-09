lm_default_config = {
    # GENERAL
    "seed": None,
    "verbose": True,
    "show_plots": True,
    # Device (default GPU)
    "device": "cpu",
    # TRAIN
    "train_config": {
        # Dataloader
        "data_loader_config": {"batch_size": 1000, "num_workers": 1},
        # Classifier
        # LF precision initializations / priors (float or np.array)
        "prec_init": 0.7,
        # Centered L2 regularization strength (int, float, or np.array)
        "l2": 0.0,
        # Optimizer
        "optimizer_config": {
            "optimizer": "sgd",
            "optimizer_common": {"lr": 0.01},
            # Optimizer - SGD
            "sgd_config": {"momentum": 0.9},
        },
        # Scheduler
        "lr_scheduler": None,
        # Train loop
        "n_epochs": 100,
        # Logger (see metal/logging/writer.py for descriptions)
        "logger": True,
        "logger_config": {
            "log_train_every": 1,  # How often train loss is reported
            "log_train_metrics": ["train/loss"],
            "log_train_metrics_func": None,
        },
    },
}

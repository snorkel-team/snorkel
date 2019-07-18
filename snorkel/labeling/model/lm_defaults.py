lm_default_config = {
    # GENERAL
    "seed": None,
    "verbose": True,
    # Device (default GPU)
    "device": "cpu",
    # TRAIN
    "train_config": {
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
            # Optimizer - RMSProp
            "rmsprop_config": {"momentum": 0.9},
            # Optimizer - Adam, SparseAdam
            "adam_config": {"betas": (0.9, 0.999)},
        },
        # Scheduler
        "lr_scheduler": None,
        "lr_scheduler_config": {
            "lr_freeze": 0,
            # Optimizer - Exponential
            "exponential_config": {"gamma": 0.9},
        },
        # Train loop
        "n_epochs": 100,
        "log_train_every": 10,
    },
}

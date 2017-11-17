config = {
    'candidate_name' : 'ChemicalDisease',
    'candidate_entities' : ['chemical', 'disease'],

    'disc_model_class': 'logreg',
    'disc_model_search_space': 10,
    'disc_init_params': {
        'n_threads': 16,
        'seed'     : 123,
    },
    'disc_params_default': {
        'lr':         0.01,
        'n_epochs':   20,
        'rebalance':  0.25,
        'batch_size': 128,
        'print_freq': 5,
    },    
    'disc_params_range': {
        'lr'        : [1e-2, 1e-3, 1e-4],
        'rebalance' : [0.25, 0.5, False],
        'n_epochs'  : [25, 50, 100],
        'batch_size': [16, 32, 64],
    },
    'disc_eval_batch_size': None,
}
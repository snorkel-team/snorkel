config = {
    'candidate_name' : 'Spouse',
    'candidate_entities' : ['person1', 'person2'],

    'babbler_label_split': 1,

    'gen_init_params': {
		'class_prior'           : False, # TRUE!?
        'lf_propensity'         : True,
    },
    'gen_params_default': {
        'decay'    : 0.99,
        'epochs'   : 50,
        'reg_param': 0.5,
        'step_size': 1e-4,
        # used iff class_prior = True
        'init_class_prior' : -1.15, # (9%)
    },

    # LSTM
    # 'disc_model_class': 'lstm',
    # 'disc_model_search_space': 10,
    # 'disc_init_params': {
    #     'n_threads': 16,
    #     'seed'     : 123,
    # },
    # 'disc_params_default': {
    #     'lr':         0.01,
    #     'dim':        50,
    #     'n_epochs':   20,
    #     'dropout':    0.5,
    #     'rebalance':  0.25,
    #     'batch_size': 128,
    #     'max_sentence_length': 100,
    #     'print_freq': 1,
    # },    
    # 'disc_params_range': {
    #     'lr'        : [1e-2, 1e-3, 1e-4],
    #     'dim'       : [64, 128],
    #     'dropout'   : [0.1, 0.25, 0.5],
    #     'rebalance' : [0.25, 0.5, False],
    # },
    # 'disc_eval_batch_size': 256,

    'disc_model_class': 'logreg',
    'disc_model_search_space': 10,
    'disc_init_params': {
        'n_threads': 16,
        'seed'     : 123,
    },
    'disc_params_default': {
        'lr':         0.01,
        'dim':        50,
        'n_epochs':   20,
        'dropout':    0.5,
        'rebalance':  0.25,
        'batch_size': 128,
        'max_sentence_length': 100,
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
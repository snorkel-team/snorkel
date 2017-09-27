config = {
    'candidate_name' : 'Spouse',
    'candidate_entities' : ['person1', 'person2'],
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
    'disc_params_default': {
        'lr':         0.01,
        'dim':        50,
        'n_epochs':   20,
        'dropout':    0.5,
        'rebalance':  0.25,
        'batch_size': 128,
        'max_sentence_length': 100,
        'print_freq': 1,
    },    
}
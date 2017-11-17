config = {
    'candidate_name' : 'ChemicalDisease',
    'candidate_entities' : ['chemical', 'disease'],

    # collect
    'babbler_label_split': 1,

    # supervise
    'gen_init_params': {
        'lf_propensity'         : True,
        'lf_prior'              : False, 
		'class_prior'           : False,
        'lf_class_propensity'   : False,
        'seed'                  : 123,
    },
    'gen_params_range': {
        'step_size'     : [1e-2, 1e-3, 1e-4, 1e-5],
        'decay'         : [0.9, 0.95, 0.99],
        'reg_param'     : [0.0, 0.01, 0.1, 0.25, 0.5, 0.75],
        'epochs'        : [10, 25, 50, 100],
    },
    'gen_params_default': {
        # Used iff lf_prior = True
        'LF_acc_prior_weight_default' : 0.5, # [0, 0.5, 1.0, 1.5] = (50%, 62%, 73%, 82%)
        # Used iff class_prior = True
        'init_class_prior' : -0.695, # (33.3%, based on dev balance)
        # logit = ln(p/(1-p)), p = exp(logit)/(1 + exp(logit))
    },
    'tune_b': True,

    # classify
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
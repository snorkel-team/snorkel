from utils import STAGES

global_config = {
    ### SETUP ###
    'domain': None,
    'log_dir': 'logs',
    'reports_dir': 'reports',
    'postgres': False,
    'parallelism': 1,
    'splits': [0, 1, 2],
    'verbose': True,
    'debug': False,
    'no_plots': False,
    'seed': 0,
    'start_at': STAGES.SETUP, # Stage of pipeline to start at
    'end_at': STAGES.ALL, # Stage of pipeline to end at (inclusive)

    ### PARSE ###
    'max_docs': None,

    ### EXTRACT ###

    ### LOAD_GOLD ###

    ### COLLECT ###

    ## Babbler
    'babbler_candidate_split': 0, # Look for explanation candidates in this split
    'babbler_label_split': 0, # Check label signatures based on this split
    'beam_width': 10,
    'top_k': -1,
    # filters
    'do_filter_duplicate_semantics': True, 
    'do_filter_consistency': True, 
    'do_filter_duplicate_signatures': True, 
    'do_filter_uniform_signatures': True,

    ### LABEL ###

    ### SUPERVISE ###
    'supervision': 'generative', # ['traditional', 'majority_vote', 'generative'],

    ## traditional
    'max_train': None,  # Number of ground truthed training candidates
    
    ## generative
    'gen_model_search_space': 10,
    'gen_f_beta': 1,
    'gen_init_params': {
		'class_prior'           : False,
        'lf_prior'              : False, 
        'lf_propensity'         : True,
        'lf_class_propensity'   : False,
        'seed'                  : 123,
    },
    'gen_params_range': {
        'step_size'                   : [1e-2, 1e-3, 1e-4, 1e-5],
        'decay'                       : [0.9, 0.95, 0.99],
        'reg_param'                   : [0.0, 0.01, 0.1, 0.25, 0.5],
        # Used iff lf_prior = True
        # 'LF_acc_prior_weights'        : [None],
        'LF_acc_prior_weight_default' : [0.5, 1.0, 1.5], # (73%, 88%, 95%)
    },
    'gen_params_default': {
    	'decay'    : 0.95,
        'epochs'   : 50,
        'reg_param': 0.1,
        # used iff class_prior = True
        'init_class_prior' : 0, # 0 = 50% pos, -1.15 = 9% pos
    },

    # dependencies
    'learn_deps': False,
    'deps_thresh': 0.01,

    ## display
    'display_accuracies': True,
    'display_learned_accuracies': False,
    'display_correlation': False,
    'display_marginals': True,

    ### CLASSIFY ###
    'disc_model_class': 'lstm',
    'disc_model_search_space': 10,
	'disc_init_params': {
        'n_threads': 16,
        'seed'     : 123,
    },
    'disc_params_range': {
        'lr'        : [1e-2, 1e-3, 1e-4],
        'dim'       : [64, 128],
        'dropout'   : [0.1, 0.25, 0.5],
        'rebalance' : [0.25, 0.5, False],
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
    'disc_eval_batch_size': 256,
    
    ## Non-GridSearch parameters
    'b': 0.5,
}



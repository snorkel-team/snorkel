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
    'babbler_label_split': 1, # Check label signatures based on this split
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
    'gen-init-params': {
		'class_prior'           : False,
        'lf_prior'              : False, 
        'lf_propensity'         : True,
        'lf_class_propensity'   : False,
        'seed'                  : 123,
    },
    'gen-params-range': {
        'step_size'                   : [1e-2, 1e-3, 1e-4, 1e-5],
        'reg_param'                   : [0.0, 0.01, 0.1, 0.5],
        'LF_acc_prior_weights'        : [None],  # only used if lf_prior=True
        'LF_acc_prior_weight_default' : [0.5, 1.0, 1.5], # only used if lf_prior=True
        'init_class_prior'            : [-1.0] # only used if class_prior=True
    },
    'gen-params-default': {
    	'decay'    : 0.95,
        'epochs'   : 50,
        'reg_param': 0.1,
    },

    # dependencies
    'learn_deps': False,
    'deps-thresh': 0.01,

    ## display
    'display_accuracies': True,
    'display_learned_accuracies': False,
    'display_correlation': False,
    'display_marginals': True,

    ### CLASSIFY ###
    'disc-model-class': 'lstm',
    'disc_model_search_space': 10,
	'disc-init-params': {
        'n_threads': 16,
        'seed'     : 123,
    },
    'disc-params-range': {
        'lr'        : [1e-2, 1e-3, 1e-4],
        'dim'       : [50, 100],
        'dropout'   : [0.1, 0.25, 0.5],
        'rebalance' : [0.1, 0.25],
    },
    'disc-params-default': {
        'lr':         0.01,
        'dim':        50,
        'n_epochs':   20,
        'dropout':    0.5,
        'rebalance':  0.25,
        'batch_size': 128,
        'max_sentence_length': 100,
        'print_freq': 1,
    },
    'disc-eval-batch-size': 256,
    
    ## Non-GridSearch parameters
    'b': 0.5,
}



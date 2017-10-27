import os

config = {
    'candidate_name' : 'Biker',
    'candidate_entities' : ['person', 'bike'],
    'splits' : [0,1],
    
    'download_data': False,
    'anns_path': os.environ['SNORKELHOME'] + '/tutorials/babble/bike/data/',
    'slim_ws_path': '/dfs/scratch0/bradenjh/slim_ws/',

    'tune_b': False,
    
    # GENERATIVE
    'gen_params_range': {
        'step_size'                   : [1e-3, 1e-5],
        'decay'                       : [0.9, 0.99],
        'reg_param'                   : [0.0, 0.01, 0.1, 0.25, 0.5],
        'epochs'                      : [50, 100, 200]
        # 'LF_acc_prior_weights'        : [None], # Used iff lf_prior = True        
    },
    'gen_params_default': {
    	'step_size': 1e-5,
        'decay'    : 0.9,
        'epochs'   : 100,
        'reg_param': 0.01,
        'LF_acc_prior_weight_default' : 0.5, # [0.5, 1.0, 1.5] = (73%, 88%, 95%) # Used iff lf_prior = True
        'init_class_prior' : 0, # 0 = 50% pos, -1.15 = 9% pos # Used iff class_prior = True
    },

    # DISCRIMINATIVE
    'disc_model_class': 'inception_v3',
    'print_freq': 1,
    'optimizer': 'adam',
    'opt_epsilon': 1.0,
    'disc_params_search': {
        'lr'        : [.01, .1, 1, 10, 100],
        'max_steps' : [100, 200, 500, 1000]
    },
}

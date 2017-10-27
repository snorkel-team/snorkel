import os

config = {
    'candidate_name' : 'Drinker',
    'candidate_entities' : ['person', 'cup'],
    'splits' : [0,1],

    'download_data': False,
    'anns_path': os.environ['SNORKELHOME'] + '/tutorials/babble/drink/data/',
    'slim_ws_path': '/dfs/scratch0/bradenjh/slim_ws',
    # 'slim_ws_path': '/Users/bradenjh/repos/slim_ws/', 

    'tune_b': False,

    # GENERATIVE
    'gen_model_search_space': 1,
    'gen_params_range': {
        'step_size'                   : [1e-5],
        'decay'                       : [0.9],
        'reg_param'                   : [0.01],
        'epochs'                      : [50]
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
    'print_freq': 5,
    'optimizer': 'adam',
    'opt_epsilon': 1.0,
    'disc_params_search': {
        'lr'        : [1, 50],
        'weight_decay': [1e-5, 1e-4, 1e-3, 1e-2],
        'max_steps' : [200, 1000],
    },   
}


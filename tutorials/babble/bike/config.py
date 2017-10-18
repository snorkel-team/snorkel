import os

config = {
    'candidate_name' : 'Biker',
    'candidate_entities' : ['person', 'bike'],
    'splits' : [0,1],
    
    'download_data': False,
    'anns_path': os.environ['SNORKELHOME'] + '/tutorials/babble/bike/data/',
    'slim_ws_path': '/Users/bradenjh/repos/slim_ws/',

    'tune_b': False,

    'disc_model_class': 'inception_v3',
    'print_freq': 1,
    'optimizer': 'adam',
    'disc_params_search': {
        'lr'        : [0.001, 0.01, 0.1, 1, 10, 100],
        'max_steps' : [10, 30, 50, 100, 200, 500]
    },
}
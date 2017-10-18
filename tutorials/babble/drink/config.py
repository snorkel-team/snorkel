import os

config = {
    'candidate_name' : 'Drinker',
    'candidate_entities' : ['person', 'cup'],
    'splits' : [0,1],

    'download_data': False,
    'anns_path': os.environ['SNORKELHOME'] + '/tutorials/babble/drink/data/',
    'slim_ws_path': '/Users/bradenjh/repos/slim_ws/',

    'disc_model_class': 'inception_v3',
    'print_freq': 1,
    'optimizer': 'adam',
    'disc_params_search': {
        'lr'        : [1, 50],
        'max_steps' : [3, 5]
    },
    
}
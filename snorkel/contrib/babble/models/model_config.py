configuration = {
    # GENERAL
    'domain': None,
    'parallelism': 1,
    'max_docs': None,
    'splits': [0, 1, 2],
    'verbose': True,
    'seed': 0,

    # SUPERVISON
    # source
    'source': 'py', # {'py', 'nl'}
    'include': ['correct', 'passing'],
    'paraphrases': False,
    # settings
    'model_dep': False,
    'majority_vote': False,
    # restrictions
    'max_lfs': None,
    'max_train': None,
    'threshold': 1.0/3.0,
    # display
    'display_correlation': False,
    'display_marginals': False,
    # real-world conditions
    'include_py_only_lfs': False,
    'remove_paren': True,
    # testing
    'traditional': False, # e.g, 1000
    'empirical_from_train': False,

    # BABBLER
    'babbler_split': 1,
    'beam_width': 10,
    'top_k': -1,
    # filters
    'do_filter_duplicate_semantics': True, 
    'do_filter_consistency': True, 
    'do_filter_duplicate_signatures': True, 
    'do_filter_uniform_signatures': True,

    # CLASSIFICATION,
    'disc_model': 'lstm',
    'num_search': 10,
    'num_epochs': 50,
    'rebalance': True,
    'b': 0.5,
    'lr': [1e-5, 1e-2],
    'l1_penalty': [1e-6, 1e-2],
    'l2_penalty': [1e-6, 1e-2],
    'print_freq': 5,
}
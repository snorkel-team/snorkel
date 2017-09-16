from utils import STAGES

config = {
    # GENERAL
    'domain': None,
    'parallelism': 1,
    'max_docs': None,
    'splits': [0, 1, 2],
    'verbose': True,
    'seed': 0,
    'start_at': STAGES.SETUP, # Stage of pipeline to start at
    'end_at': STAGES.ALL, # Stage of pipeline to end at (inclusive)

    # BABBLER
    'babbler_candidate_split': 0, # Look for explanation candidates in this split
    'babbler_label_split': 1, # Check label signatures based on this split
    'beam_width': 10,
    'top_k': -1,
    # filters
    'do_filter_duplicate_semantics': True, 
    'do_filter_consistency': True, 
    'do_filter_duplicate_signatures': True, 
    'do_filter_uniform_signatures': True,

    # SUPERVISON
    # settings
    'traditional': False, # e.g, 1000
    'majority_vote': False,
    'learn_deps': False,
    # gen model paramaters
    'epochs': 100,
    'decay': None, # 0.95 OR 0.001 * (1.0 /epochs),
    'step_size': None, # 0.005 OR 0.1/L_train.shape[0],
    'reg_param': 1e-6, #1e-3?
    # restrictions
    'max_lfs': None,
    'max_train': None,
    'threshold': 1.0/3.0,
    # display
    'display_accuracies': True,
    'display_learned_accuracies': True,
    'display_correlation': False,
    'display_marginals': True,
    # real-world conditions
    'remove_paren': True,

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
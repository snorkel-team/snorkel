import argparse
from imp import load_source
import os

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def expand_dicts(args):
    """
    Expand flags that correspond to values in dictionaries in the config file 
    so that they match the structure of the config file.
    """    
    for k, v in args.items():
        if ':' in k:
            group, var = k.split(':')
            if group not in args:
                args[group] = {}
            args[group][var] = v
            del args[k]
    return args


if __name__ == '__main__':
    """
    This launch script exists primarily to add a flag interface for launching
    pipeline.run(). All flags correspond to values in the global_config file.
    Documentation and default values for individual config values should be 
    stored in global_config, not here. Unusued flags will not overwrite the
    values in config.
    """

    # Parse command-line args
    argparser = argparse.ArgumentParser(description="Run SnorkelPipeline object.")
    
    DOMAINS = ['spouse', 'cdr', 'bike', 'drink', 'stub']
    argparser.add_argument('--domain', type=str, default='stub', choices=DOMAINS,
        help="Name of experiment subdirectory in tutorials/babble/")

    # Control flow args
    argparser.add_argument('--start_at', type=int)
    argparser.add_argument('--end_at', type=int)

    # Babble args
    argparser.add_argument('--max_explanations', type=int)
    argparser.add_argument('--apply_filters', type=str2bool)

    # Supervision args
    SUPERVISION = ['traditional', 'majority', 'generative']
    argparser.add_argument('--supervision', type=str, choices=SUPERVISION)
    argparser.add_argument('--max_train', type=int)
    argparser.add_argument('--learn_deps', type=str2bool)
    argparser.add_argument('--deps_thresh', type=float)
    ## model args
    argparser.add_argument('--gen_init_params:class_prior', type=str2bool)
    argparser.add_argument('--gen_init_params:lf_prior', type=str2bool)
    argparser.add_argument('--gen_init_params:lf_propensity', type=str2bool)
    argparser.add_argument('--gen_init_params:lf_class_propensity', type=str2bool)
    ## hyperparameters

    # Classify args
    argparser.add_argument('--disc_model_class', type=str)

    argparser.add_argument('--disc_params_default:batch_size', type=int)
    argparser.add_argument('--disc_params_default:n_epochs', type=int)
    argparser.add_argument('--disc_params_default:lr', type=float)
    argparser.add_argument('--disc_params_default:rebalance', type=float)
    
    argparser.add_argument('--disc_params_range:batch_size', type=int, action='append')
    argparser.add_argument('--disc_params_range:n_epochs', type=int, action='append')
    argparser.add_argument('--disc_params_range:lr', type=float, action='append')
    argparser.add_argument('--disc_params_range:rebalance', type=float, action='append')

    # Search
    argparser.add_argument('--seed', type=int)
    argparser.add_argument('--gen_model_search_space', type=int)
    argparser.add_argument('--disc_model_search_space', type=int)

    # Scaling args
    argparser.add_argument('--max_docs', type=int,
        help="""[Deprecated] Maximum documents to parse;
        NOTE: This will also filter dev and test docs. 
        See --training_docs to limit just training docs.""")
    argparser.add_argument('--debug', action='store_true',
        help="""Reduces max_docs, grid search sizes, and num_epochs""")        

    # Logging
    argparser.add_argument('--reports_dir', type=str)

    # Data
    argparser.add_argument('--download_data', action='store_true')

    # Display args    
    argparser.add_argument('--verbose', action='store_true')
    argparser.add_argument('--no_plots', action='store_true')

    # DB configuration args
    argparser.add_argument('--db_name', type=str, default=None,
        help="Name of the database; defaults to babble_{domain}")
    argparser.add_argument('--db_port', type=str, default=None)
    argparser.add_argument('--postgres', action='store_true')
    argparser.add_argument('--parallelism', type=int)

    # Parse arguments
    args = vars(argparser.parse_args())
    args = expand_dicts(args)

    # Get the DB connection string and add to globals
    default_db_name = 'babble_' + args['domain'] + ('_debug' if args['debug'] else '')
    DB_NAME = args['db_name'] if args['db_name'] is not None else default_db_name
    if not args['postgres']:
        DB_NAME += ".db"
    DB_TYPE = "postgres" if args['postgres'] else "sqlite"
    DB_ADDR = "localhost:{0}".format(args['db_port']) if args['db_port'] else ""
    os.environ['SNORKELDB'] = '{0}://{1}/{2}'.format(DB_TYPE, DB_ADDR, DB_NAME)
    print("$SNORKELDB = {0}".format(os.environ['SNORKELDB']))

    # All Snorkel imports must happen after $SNORKELDB is set
    from snorkel import SnorkelSession
    from snorkel.models import candidate_subclass

    from config import global_config
    from config_utils import get_local_pipeline, merge_configs

    # Resolve config conflicts (args > local config > global config)
    config = merge_configs(args)
    if args['verbose'] > 0:
        print(config)

    # Create session
    session = SnorkelSession()

    # Create candidate_class
    candidate_class = candidate_subclass(config['candidate_name'], 
                                         config['candidate_entities'])

    # Create pipeline 
    pipeline = get_local_pipeline(args['domain'])
    pipe = pipeline(session, candidate_class, config)

    # Run!
    pipe.run()

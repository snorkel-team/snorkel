import argparse
from imp import load_source
import os

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

    # Scaling args
    argparser.add_argument('--max_docs', type=int,
        help="""[Deprecated] Maximum documents to parse;
        NOTE: This will also filter dev and test docs. 
        See --training_docs to limit just training docs.""")
    argparser.add_argument('--debug', action='store_true',
        help="""Reduces max_docs, grid search sizes, and num_epochs""")        

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
    args = argparser.parse_args()
    if args.verbose:
        print(args)

    # Get the DB connection string and add to globals
    default_db_name = 'babble_' + args.domain + ('_debug' if args.debug else '')
    DB_NAME = args.db_name if args.db_name is not None else default_db_name
    if not args.postgres:
        DB_NAME += ".db"
    DB_TYPE = "postgres" if args.postgres else "sqlite"
    DB_ADDR = "localhost:{0}".format(args.db_port) if args.db_port else ""
    os.environ['SNORKELDB'] = '{0}://{1}/{2}'.format(DB_TYPE, DB_ADDR, DB_NAME)
    print("$SNORKELDB = {0}".format(os.environ['SNORKELDB']))

    # All Snorkel imports must happen after $SNORKELDB is set
    from snorkel import SnorkelSession
    from snorkel.models import candidate_subclass

    from config import global_config
    from config_utils import recursive_merge_dicts, get_local_config, get_local_pipeline

    # Resolve config conflicts (args > local config > global config)
    local_config = get_local_config(args.domain)
    config = recursive_merge_dicts(global_config, local_config)
    config = recursive_merge_dicts(config, vars(args))
    if args.verbose > 0:
        print(config)

    # Create session
    session = SnorkelSession()

    # Create candidate_class
    candidate_class = candidate_subclass(config['candidate_name'], 
                                         config['candidate_entities'])

    # Create pipeline 
    pipeline = get_local_pipeline(args.domain)
    pipe = pipeline(session, candidate_class, config)

    # Run!
    pipe.run()
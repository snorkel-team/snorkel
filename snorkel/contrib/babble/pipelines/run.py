import argparse
from imp import load_source
import os

from snorkel import SnorkelSession
from snorkel.models import candidate_subclass

from config import global_config
from utils import recursive_merge_dicts, get_local_config, get_local_pipeline

if __name__ == '__main__':

    # Parse command-line args
    argparser = argparse.ArgumentParser(description="Run SnorkelPipeline object.")
    
    EXPS = ['spouse', 'cdr', 'bike', 'drink']
    argparser.add_argument('--domain', type=str, default='stub', choices=EXPS,
        help="Name of experiment subdirectory in tutorials/babble/")

    # Control flow args
    argparser.add_argument('--start_at', type=int)
    argparser.add_argument('--end_at', type=int)

    # Display args    
    argparser.add_argument('--verbose', action='store_true')

    # DB configuration args
    argparser.add_argument('--db_name', type=str, default=None,
        help="Name of the database; defaults to snorkel_{exp}")
    argparser.add_argument('--db_port', type=str, default=None)
    argparser.add_argument('--postgres', action='store_true')

    # Parse arguments
    args = argparser.parse_args()
    if args.verbose:
        print(args)

    # Resolve config conflicts (args > local config > global config)
    local_config = get_local_config(args.domain)
    config = recursive_merge_dicts(global_config, local_config)
    config = recursive_merge_dicts(config, vars(args))
    if args.verbose > 0:
        print(config)

    # Get the DB connection string and add to globals
    DB_NAME = "babble_" + args.domain if args.db_name is None else args.db_name
    if not args.postgres:
        DB_NAME += ".db"
    DB_TYPE = "postgres" if args.postgres else "sqlite"
    DB_ADDR = "localhost:{0}".format(args.db_port) if args.db_port else ""
    os.environ['SNORKELDB'] = '{0}://{1}/{2}'.format(DB_TYPE, DB_ADDR, DB_NAME)
    print("$SNORKELDB = {0}".format(os.environ['SNORKELDB']))

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
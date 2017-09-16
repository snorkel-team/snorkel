from imp import load_source
import os
from time import time

class STAGES:
    SETUP = 0
    PARSE = 1
    EXTRACT = 2
    LOAD_GOLD = 3
    COLLECT = 4
    LABEL = 5
    SUPERVISE = 6
    CLASSIFY = 7
    ALL = 10


class PrintTimer:
    """Prints msg at start, total time taken at end."""
    def __init__(self, msg, prefix="###"):
        self.msg = msg
        self.prefix = prefix + " " if len(prefix) > 0 else prefix

    def __enter__(self):
        self.t0 = time()
        print("{0}{1}".format(self.prefix, self.msg))

    def __exit__(self, type, value, traceback):
        print ("{0}Done in {1:.1f}s.\n".format(self.prefix, time() - self.t0))


def recursive_merge_dicts(x, y):
    """
    Merge dictionary y into x, overwriting elements of x when there is a
    conflict, except if the element is a dictionary, in which case recurse.
    """
    for k, v in y.iteritems():
        if k in x and isinstance(x[k], dict):
            x[k] = recursive_merge_dicts(x[k], v)
        else:
            x[k] = v
    return x


def get_local_config(domain):
    local_config_path = os.path.join(os.environ['SNORKELHOME'], 
        'tutorials', 'babble', domain, 'config.py')
    if not os.path.exists(local_config_path):
        raise Exception("The config.py for the {} domain was not found at {}.".format(
            domain, local_config_path))
    local_config = load_source('local_config', local_config_path)
    return local_config.config


def get_local_pipeline(domain):
    pipeline_path = os.path.join(os.environ['SNORKELHOME'],
        'tutorials', 'babble', domain, '{}_pipeline.py'.format(domain))
    if not os.path.exists(pipeline_path):
        raise Exception("Pipeline for the {} domain ({}) was not found at {}.".format(
            domain, pipeline_name, pipeline_path))
    pipeline_module = load_source('pipeline_module', pipeline_path)
    pipeline_name = '{}Pipeline'.format(domain.capitalize())
    pipeline = getattr(pipeline_module, pipeline_name)
    return pipeline
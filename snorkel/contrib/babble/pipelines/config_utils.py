from imp import load_source
import os

from config import global_config

def merge_configs(config):
    if 'domain' not in config:
        raise Exception("config must have non-None value for 'domain'.")
    local_config = get_local_config(config['domain'])
    config = recursive_merge_dicts(local_config, config)
    config = recursive_merge_dicts(global_config, config)
    return config  


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
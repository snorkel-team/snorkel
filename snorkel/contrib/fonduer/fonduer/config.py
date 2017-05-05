import os
import yaml


def dict2obj(d):
    if isinstance(d, dict):
        n = {}
        for item in d:
            if isinstance(d[item], dict):
                n[item] = dict2obj(d[item])
            elif isinstance(d[item], (list, tuple)):
                n[item] = [dict2obj(i) for i in d[item]]
            else:
                n[item] = d[item]
        return type('obj_from_dict', (object,), n)
    else:
        return d


with open(os.environ.get('FONDUERHOME', '') + "/" + "settings.yaml", "r") as f:
    settings = dict2obj(yaml.load(f))

import json
import os
from subprocess import check_output
from time import time, strftime

import numpy as np
from pandas import DataFrame, Series

class STAGES:
    SETUP = 0
    PARSE = 1
    EXTRACT = 2
    LOAD_GOLD = 3
    FEATURIZE = 4
    COLLECT = 5
    LABEL = 6
    SUPERVISE = 7
    CLASSIFY = 8
    ALL = 10

### LOGGING
def git_commit_hash(path=None):
    # A bit of a hack for path is not None...
    if path is not None:
        original_path = os.getcwd()
        os.chdir(path)
    h = check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()
    if path is not None:
        os.chdir(original_path)
    return h


### REPORTING TOOLS
def final_report(config, scores):
    # Print and save final score report
    ks = list(scores.keys())
    cols = ['Precision', 'Recall', 'F1 Score']
    d = {
        'Precision' : Series(data=[scores[k][0] for k in ks], index=ks),
        'Recall'    : Series(data=[scores[k][1] for k in ks], index=ks),
        'F1 Score'  : Series(data=[scores[k][2] for k in ks], index=ks),
    }
    df = DataFrame(data=d, index=ks)
    print(df)

    # Assemble the report, to be saved as a json file
    df_scores = df.to_dict()
    report = {
        'snorkel-commit': git_commit_hash(path=os.environ['SNORKELHOME']),
        'scores': df_scores,
        'config': config,
    }

    # Save to file
    report_dir = os.path.join(os.environ['SNORKELHOME'], config['reports_dir'], strftime("%Y_%m_%d"))
    report_name = '{0}_{1}.json'.format(config['domain'], strftime("%H_%M_%S"))
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    with open(os.path.join(report_dir, report_name), 'wb') as f:
        json.dump(report, f, indent=2)


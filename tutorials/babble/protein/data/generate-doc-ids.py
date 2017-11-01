import numpy as np
from collections import defaultdict
import os
# import pickle
from six.moves.cPickle import dump

# TODO rename all_ids to unexplained_ids
def pkr_doc_split(
    id_fname="./all_ids.txt",
    unlabeled_id_fname="./unlabelled_ids.txt",
    out_fname='./all_pkr_ids.pkl',
    seed=1701
):
    id_dict = defaultdict(list)
    explainedIds = [
            '28582909',
            '28582849',
            '28582834'
            ]
    unexplainedIds = []
    with open(id_fname, 'rt') as f:
        for line in f:
            unexplainedIds.append(line.strip('\n'))
    unlabeledIds= []
    with open(unlabeled_id_fname, 'rt') as f:
        for line in f:
            unlabeledIds.append(line.strip('\n'))
    np.random.seed(seed)
    np.random.shuffle(unexplainedIds)
    id_dict = {
            'train': unlabeledIds,
            'dev': explainedIds + unexplainedIds[0:22],
            'test': unexplainedIds[22:50]
        }
    print("id_dict:",id_dict)

    with open(out_fname, 'w') as f:
        dump((id_dict['train'], id_dict['dev'], id_dict['test']), f)
if __name__ == '__main__':
    pkr_doc_split()

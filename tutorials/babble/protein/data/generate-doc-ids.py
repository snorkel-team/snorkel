import numpy as np
from collections import defaultdict
import os
# import pickle
from six.moves.cPickle import dump

# TODO rename all_ids to unexplained_ids
def pkr_doc_split(
    labeled_id_fname="./labeled_ids.txt",
    all_id_fname="./all_ids.txt",
    out_fname='./all_pkr_ids.pkl',
    seed=1701
):
    id_dict = defaultdict(list)
    explainedIds = [
            '28582909',
            '28582849',
            '28582834'
            ]
    labeledIds= []
    with open(labeled_id_fname, 'rt') as f:
        for line in f:
            labeledIds.append(line.strip('\n'))
    allIds= []
    with open(all_id_fname, 'rt') as f:
        for line in f:
            allIds.append(line.strip('\n'))
    unlabeledIds = [val for val in allIds if val not in labeledIds] 
    np.random.seed(seed)
    np.random.shuffle(labeledIds)
    print('unlabeled length:',len(unlabeledIds))
    print('explained length:',len(explainedIds))
    print('lableled length:',len(labeledIds))
    id_dict = {
            'train': unlabeledIds,
            'dev': explainedIds + labeledIds[0:255],
            'test': labeledIds[255:517]
        }
    print("id_dict:",id_dict)
    print("train dev test:",len(id_dict['train']),len(id_dict['dev']),len(id_dict['test']))

    with open(out_fname, 'w') as f:
        dump((id_dict['train'], id_dict['dev'], id_dict['test']), f)
if __name__ == '__main__':
    pkr_doc_split()

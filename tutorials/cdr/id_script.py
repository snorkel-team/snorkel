import numpy as np
from collections import defaultdict
from six.moves.cPickle import dump

def cdr_doc_split(
    train_fname='data/CDR_TrainingSet.BioC.xml',
    dev_fname='data/CDR_DevelopmentSet.BioC.xml',
    test_fname='data/CDR_TestSet.BioC.xml',
    out_fname='data/doc_ids.pkl', n_dev=100, seed=1701,
):
    # Get IDs from each file
    docs = {'train': train_fname, 'dev': dev_fname, 'test': test_fname}
    id_dict = defaultdict(list)
    for set_name, fname in docs.iteritems():
        with open(fname, 'rb') as f:
            for line in f:
                if line.strip() == '<document>':
                    line = next(f)
                    doc_id = line[4:(-6 - (set_name == 'test'))]
                    id_dict[set_name].append(doc_id)
        print("Found {0} documents in {1}. Sample: {2}.".format(
            len(id_dict[set_name]), set_name, doc_id
        ))
    # Add part of dev to train
    n_dev = min(n_dev, 500)
    np.random.seed(seed)
    np.random.shuffle(id_dict['dev'])
    id_dict['train'].extend(id_dict['dev'][n_dev:])
    id_dict['dev'] = id_dict['dev'][:n_dev]
    print("New sizes: {0}".format(' '.join(
    	'{0}={1}'.format(k, len(v)) for k, v in id_dict.items()
    )))
    # Save to file
    with open(out_fname, 'wb') as f:
        dump((id_dict['train'], id_dict['dev'], id_dict['test']), f)

if __name__ == '__main__':
	cdr_doc_split()

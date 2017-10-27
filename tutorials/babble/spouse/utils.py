import os

import pandas as pd

from snorkel.models import StableLabel
from snorkel.db_helpers import reload_annotator_labels

FPATH = os.environ['SNORKELHOME'] + '/tutorials/babble/spouse/data/gold_labels.tsv'

def load_external_labels(session, candidate_class, annotator_name='gold', path=FPATH, splits=[1,2]):
    gold_labels = pd.read_csv(path, sep="\t")
    for index, row in gold_labels.iterrows():    

        # We check if the label already exists, in case this cell was already executed
        context_stable_ids = "~~".join([row['person1'], row['person2']])
        query = session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_ids)
        query = query.filter(StableLabel.annotator_name == annotator_name)
        if query.count() == 0:
            session.add(StableLabel(
                context_stable_ids=context_stable_ids,
                annotator_name=annotator_name,
                value=row['label']))
                    
        # Because it's a symmetric relation, load both directions...
        context_stable_ids = "~~".join([row['person2'], row['person1']])
        query = session.query(StableLabel).filter(StableLabel.context_stable_ids == context_stable_ids)
        query = query.filter(StableLabel.annotator_name == annotator_name)
        if query.count() == 0:
            session.add(StableLabel(
                context_stable_ids=context_stable_ids,
                annotator_name=annotator_name,
                value=row['label']))

    # Commit session
    session.commit()

    # Reload annotator labels
    splits = splits if isinstance(splits, list) else [splits]
    for split in splits:
        reload_annotator_labels(session, 
                                candidate_class, 
                                annotator_name, 
                                split=split, 
                                filter_label_split=False)

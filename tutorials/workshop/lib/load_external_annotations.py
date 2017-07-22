import pandas as pd
from snorkel.models import StableLabel
from snorkel.db_helpers import reload_annotator_labels

FPATH = 'data/gold_labels.tsv'

def load_external_labels(session, candidate_class, annotator_name='gold'):
    gold_labels = pd.read_csv(FPATH, sep="\t")
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
    reload_annotator_labels(session, candidate_class, annotator_name, split=1, filter_label_split=False)
    reload_annotator_labels(session, candidate_class, annotator_name, split=2, filter_label_split=False)

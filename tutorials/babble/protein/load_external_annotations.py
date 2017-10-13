from collections import defaultdict
import csv
import os

from snorkel.db_helpers import reload_annotator_labels
from snorkel.models import StableLabel

DATA_ROOT = os.environ['SNORKELHOME'] + '/tutorials/babble/protein/data/'

def load_external_labels(session, candidate_class, split, annotator='gold',
    label_fname='razor10-6-17-influences-dump.csv'):
    # Load document-level relation annotations
    with open(DATA_ROOT + label_fname, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        csvreader.next()
        positives_by_doc = defaultdict(set)
        for i, row in enumerate(csvreader):
            try:
                (doc_id, text, relation, from_entity, from_start, from_end, 
                 to_entity, to_start, to_end) = row
            except:
                print("Malformed row {}.".format(i + 2))
                continue
            label = 1 if relation == 'influences' else -1
            if label == 1:
                from_stable_id = "{}::span:{}:{}".format(doc_id, from_start, from_end)
                to_stable_id = "{}::span:{}:{}".format(doc_id, to_start, to_end)
                protein_stable_id = from_stable_id if from_entity == 'Protein' else to_stable_id
                kinase_stable_id = from_stable_id if from_entity == 'Kinase' else to_stable_id
                candidate_stable_id = '~~'.join([protein_stable_id, kinase_stable_id])
                positives_by_doc[doc_id].add(candidate_stable_id)
            
    # Get split candidates
    candidates = session.query(candidate_class).filter(
        candidate_class.split == split
    ).all()
    for c in candidates:
        # Get the label by mapping document annotations to mentions
        doc_id = c.get_parent().get_parent().name
        doc_positives = positives_by_doc.get(doc_id, set())
        context_stable_ids = c.get_stable_id()
        label = 2 * int(context_stable_ids in doc_positives) - 1        
        # Get stable ids and check to see if label already exits
        query = session.query(StableLabel).filter(
            StableLabel.context_stable_ids == context_stable_ids
        )
        query = query.filter(StableLabel.annotator_name == annotator)
        # If does not already exist, add label
        if query.count() == 0:
            session.add(StableLabel(
                context_stable_ids=context_stable_ids,
                annotator_name=annotator,
                value=label
            ))

    # Commit session
    session.commit()

    # Reload annotator labels
    reload_annotator_labels(session, candidate_class, annotator,
                            split=split, filter_label_split=False)


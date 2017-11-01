from collections import defaultdict
import csv
import os

from snorkel.db_helpers import reload_annotator_labels
from snorkel.models import StableLabel

DATA_ROOT = os.environ['SNORKELHOME'] + '/tutorials/babble/protein/data/'

def load_external_labels(session, candidate_class, split, annotator='gold',
    label_fname='candidates-10-16-17.csv'):
    # Load document-level relation annotations
    with open(DATA_ROOT + label_fname, 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        csvreader.next()
        positives_by_doc = defaultdict(set)
        all_labeled_ids = []
        for i, row in enumerate(csvreader):
            try:
                (cand_id, from_influences_to, to_influences_from,
                        context_stable_id ) = row
                [from_stable_id, to_stable_id ] = context_stable_id.split("~~")
                doc_id = from_stable_id.split(':')[0]
                all_labeled_ids.append(context_stable_id)
            except:
                print("Malformed row {}.".format(i + 2))
                continue
            #proteins are always first, kinases last
            #indicates protein influences kinase
            label = 1 if (from_influences_to == 'true') else -1
            if label == 1:
                candidate_stable_id = '~~'.join([from_stable_id, to_stable_id])
                positives_by_doc[doc_id].add(candidate_stable_id)
            # TODO to_influences_from indicates the kinase influences the
            # protein.  Add support for candidate extraction and loading of
            # external relations in the other direction

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
        if query.count() == 0 and context_stable_ids in all_labeled_ids:
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


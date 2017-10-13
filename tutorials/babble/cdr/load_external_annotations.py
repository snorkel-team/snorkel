import os
from six.moves.cPickle import load

from snorkel.db_helpers import reload_annotator_labels
from snorkel.models import StableLabel

DATA_ROOT = os.environ['SNORKELHOME'] + '/tutorials/babble/cdr/data/'

def load_external_labels(session, candidate_class, split, annotator='gold',
    label_fname='cdr_relations_gold.pkl', id_fname='doc_ids.pkl'):
    # Load document-level relation annotations
    with open(DATA_ROOT + label_fname, 'rb') as f:
        relations = load(f)
    # Get split candidates
    candidates = session.query(candidate_class).filter(
        candidate_class.split == split
    ).all()
    for c in candidates:
        # Get the label by mapping document annotations to mentions
        doc_relations = relations.get(c.get_parent().get_parent().name, set())
        label = 2 * int(c.get_cids() in doc_relations) - 1        
        # Get stable ids and check to see if label already exits
        context_stable_ids = '~~'.join(x.get_stable_id() for x in c)
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

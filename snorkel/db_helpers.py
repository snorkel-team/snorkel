from .models import StableLabel, Label, Context
from .queries import get_or_create_single_key_set
from sqlalchemy.orm import object_session


def reload_annotator_labels(session, candidate_class, annotator_name):
    """
    Reloads stable annotator labels into the Label table
    Note that this function currently *does not* handle creating necessary Contexts / Candidates.
    """
    aks, ak = get_or_create_single_key_set(session, annotator_name)
    
    labels = []
    missed = []
    for al in session.query(StableLabel).filter(StableLabel.annotator_name == annotator_name).all():

        # Check for labeled Contexts
        contexts = []
        for stable_id in al.context_stable_ids:
            context = session.query(Context).filter(Context.stable_id == stable_id).first()
            if context:
                contexts.append(context)
        if len(contexts) < len(al.context_stable_ids):
            missed.append(al)
            continue

        # Check for Candidate
        candidate_query = session.query(candidate_class)
        for i, arg in enumerate(candidate_class.__argnames__):
            candidate_query = candidate_query.filter(getattr(candidate_class, arg) == contexts[i])
        candidate = candidate_query.first()
        if candidate is None:
            missed.append(al)
            continue

        # Check for Label, otherwise create
        label = session.query(Label).filter(Label.key == ak).filter(Label.candidate == candidate).first()
        if label is None:
            label = Label(candidate=candidate, key=ak, value=al.value)
            session.add(label)
            labels.append(label)

    session.commit()
    print "Labels created: %s\nLabels skipped: %s" % (len(labels), len(missed))
    return labels, missed

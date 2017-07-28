from .models import StableLabel, GoldLabel, Context, GoldLabelKey
from six import iteritems
from sqlalchemy.orm import object_session

def reload_annotator_labels(session, candidate_class, annotator_name, split, filter_label_split=True, create_missing_cands=False):
    """Reloads stable annotator labels into the AnnotatorLabel table"""
    # Sets up the AnnotatorLabelKey to use
    ak = session.query(GoldLabelKey).filter(GoldLabelKey.name == annotator_name).first()
    if ak is None:
        ak = GoldLabelKey(name=annotator_name)
        session.add(ak)
        session.commit()

    labels = []
    missed = []
    sl_query = session.query(StableLabel).filter(StableLabel.annotator_name == annotator_name)
    sl_query = sl_query.filter(StableLabel.split == split) if filter_label_split else sl_query
    for sl in sl_query.all():
        context_stable_ids = sl.context_stable_ids.split('~~')

        # Check for labeled Contexts
        # TODO: Does not create the Contexts if they do not yet exist!
        contexts = []
        for stable_id in context_stable_ids:
            context = session.query(Context).filter(Context.stable_id == stable_id).first()
            if context:
                contexts.append(context)
        if len(contexts) < len(context_stable_ids):
            missed.append(sl)
            continue

        # Check for Candidate
        # Assemble candidate arguments
        candidate_args  = {'split' : split}
        for i, arg_name in enumerate(candidate_class.__argnames__):
            candidate_args[arg_name] = contexts[i]

        # Assemble query and check
        candidate_query = session.query(candidate_class)
        for k, v in iteritems(candidate_args):
            candidate_query = candidate_query.filter(getattr(candidate_class, k) == v)
        candidate = candidate_query.first()

        # Optionally construct missing candidates
        if candidate is None and create_missing_cands:
            candidate = candidate_class(**candidate_args)

        # If candidate is none, mark as missed and continue
        if candidate is None:
            missed.append(sl)
            continue

        # Check for AnnotatorLabel, otherwise create
        label = session.query(GoldLabel).filter(GoldLabel.key == ak).filter(GoldLabel.candidate == candidate).first()
        if label is None:
            label = GoldLabel(candidate=candidate, key=ak, value=sl.value)
            session.add(label)
            labels.append(label)

    session.commit()
    print("AnnotatorLabels created: %s" % (len(labels),))

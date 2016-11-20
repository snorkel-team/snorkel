from .models import Corpus, CandidateSet, AnnotationKeySet, ParameterSet, AnnotatorLabel, Label
from .queries import get_or_create_single_key_set
from sqlalchemy.orm import object_session


# TODO
def reload_annotator_labels(session, candidate_class, context_classes, annotator_name):
    """Reloads stable annotator labels into the Label table"""
    aks, ak = get_or_create_single_key_set(annotator_name)

    for al in session.query(AnnotatorLabel).filter(AnnotatorLabel.annotator_name == annotator_name).all():

        # Check for labeled Contexts
        contexts = []
        for stable_id in al.context_stable_ids:
            context = session.query(Context).filter(Context.stable_id == stable_id).first()

            # If context does not exist, create it
            # TODO: This is a little bit involved...
            if context is None:
                pass  # TODO


def cascade_delete_set(s):
    """Given a many-to-many set, delete all contained / dependent elements"""
    session = object_session(s)

    # If a Corpus, delete all the documents and everything else cascades
    # Corpus -> Document -> Sentence -> Span -> Candidate -> Annotation
    if isinstance(s, Corpus):
        for document in s:
            session.delete(document)

    # If a CandidateSet, delete all the candidates and similarly everything will cascade
    elif isinstance(s, CandidateSet):
        for candidate in s:
            session.delete(candidate)

    # If an AnnotationKeySet, delete all AnnotationKeys, cascades down to Annotations, Parameters
    elif isinstance(s, AnnotationKeySet):
        for ak in s.keys:
            session.delete(ak)

    # ParameterSets already cascade to deleting all Parameters
    elif isinstance(s, ParameterSet):
        pass
    else:
        raise ValueError("Unhandled set type: " + s.__name__)

    # Finally, delete the set and commit
    session.delete(s)
    session.commit()

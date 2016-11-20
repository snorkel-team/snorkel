from .models import Corpus, Document, Sentence
from sqlalchemy.orm import object_session


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

    # TODO: AnnotationKeySet

    # Finally, delete the set and commit
    session.delete(s)
    session.commit()

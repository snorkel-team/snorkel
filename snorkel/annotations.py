import numpy as np
from pandas import DataFrame
from collections import defaultdict
import scipy.sparse as sparse
from .models import Candidate, CandidateSet, Feature


class CandidateAnnotator(object):
    """
    Abstract class for annotating candidates, saving these annotations to DB, and then reloading them as
    sparse matrices; generic operation which covers Annotation subclasses:
        * Feature
        * Label
    E.g. for features, LF labels, human annotator labels, etc.
    """
    def __init__(self, annotation=Annotation, key=AnnotationKey):
        self.annotation = annotation
        self.key        = key

    def create(self, candidates, annotation_generator, session, annotation_keyset):
        """
        Given a set of candidates and a generator which, given a candidate, yields annotations as key name, value
        pairs, persist these annotations in the session.

        If annotation_keyset exists already, only keep those annotations with keys in this set; otherwise,
        create and assign to the new annotation key set.
        """
        # If annotarion_key_set refers to an existing AnnotationKeySet, then use this, and only create annotations
        # which fall within this keyset
        keyset = session.query(AnnotationKeySet).filter(AnnotationKeySet.name == annotation_keyset).first()
        if keyset is not None:
            key_names = frozenset(k.name for k in keyset.keys)

        # Otherwise, create a new keyset
        else:
            keyset    = AnnotationKeySet(name=annotation_keyset)
            key_names = None

        # Create annotations (avoiding potential duplicates) and add to session
        seen = set()
        for candidate in candidates:
            seen.clear()
            for key_name, value in annotation_generator(candidate):
                if key_names is None or key_name in key_names:
                    key = self.key(name=key_name)
                    a = self.annotation(candidate=candidate, key=key, value=value)
                    if a not in seen:
                        session.add(a)
                        seen.add(a)
        session.commit()
    
    # TODO
    def add(self, candidates, annotation_generator, session, annotation_keyset):
        raise NotImplementedError()

    # TODO
    def remove(self, candidates, key_name, session, annotation_keyset):
        raise NotImplementedError()

    # TODO
    def update(self, candidates, annotation_generator, session, annotation_keyset):
        raise NotImplementedError()

    # TODO
    def load(self, candidates, session, annotation_keyset):
        raise NotImplementedError()

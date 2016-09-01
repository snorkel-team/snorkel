import numpy as np
from pandas import DataFrame
from collections import defaultdict
import scipy.sparse as sparse
from .models import Label, Feature, AnnotationKey, AnnotationKeySet, Candidate, CandidateSet
from types import GeneratorType
from .utils import get_ORM_instance


class CandidateAnnotator(object):
    """
    Abstract class for annotating candidates, saving these annotations to DB, and then reloading them as
    sparse matrices; generic operation which covers Annotation subclasses:
        * Feature
        * Label
    E.g. for features, LF labels, human annotator labels, etc.
    """
    def __init__(self, annotation=None):
        self.annotation = annotation

    def create(self, candidate_set, annotation_fns, session, annotation_key_set_name):
        # TODO: Rewrite docstring
        """
        Given a set of candidates and a list of annotation functions each of which, given a candidate,
        returns annotations as key name, value pairs: persist these annotations in the session.

        If annotation_key_set_name exists already, only keep those annotations with keys in this set; otherwise,
        create and assign to the new annotation key set.
        """
        candidate_set = get_ORM_instance(CandidateSet, session, candidate_set)
        key_set       = get_ORM_instance(AnnotationKeySet, session, key_set)

        # If annotarion_key_set refers to an existing AnnotationKeySet, then use this, and only create annotations
        # which fall within this key_set; otherwise we create a new key set
        new = False
        if key_set is None:
            print "Creating new key set..."
            new     = True
            key_set = AnnotationKeySet(name=annotation_key_set_name)
            session.add(key_set)
            session.commit()
        else:
            print "Using existing key set with %s keys" % len(key_set)

        # Create annotations (avoiding potential duplicates) and add to session
        seen_key_names = set()
        for candidate in candidate_set:
            seen_key_names.clear()
            for f in annotation_fns:

                # Handle both single-return and generator functions
                outputs = f(candidate)
                outputs = outputs if isinstance(outputs, GeneratorType) else [outputs]
                for output in outputs:

                    # An annotation function either yields a key, value tuple, or else just a value
                    # In the latter case, we use its name as the key
                    key_name, value = output if isinstance(output, tuple) else f.__name__, output

                    # Note: we also only store unique non-zero values!
                    if key_name not in seen_key_names and value != 0:
                        seen_key_names.add(key_name)
                        
                        # Get or create AnnotationKey
                        if key_name in key_set.keys:
                            key = key_set.keys[key_name]
                        elif new:
                            key = AnnotationKey(name=key_name)
                            session.add(key)
                            key_set.append(key)
                            session.commit()
                        else:
                            continue
                        session.add(self.annotation(candidate=candidate, key=key, value=value))
        session.commit()
    
    def load(self, candidate_set, key_set, session):
        # TODO: Docstring!!
        candidate_set = get_ORM_instance(CandidateSet, session, candidate_set)
        key_set       = get_ORM_instance(AnnotationKeySet, session, key_set)

        # Create sparse matrix in LIL format for incremental construction
        X = sparse.lil_matrix((len(candidate_set), len(key_set)))

        # We map on-the-fly from *ordered* but potentially non-contiguous integer ids to row/col indices
        row_index = {}
        col_index = {}

        # Construct the query
        q = session.query(Label.candidate_id, Label.key_id, Label.value).join(Candidate, AnnotationKey)
        q = q.filter(Candidate.sets.contains(candidate_set)).filter(AnnotationKey.sets.contains(key_set))
        q = q.order_by(Label.candidate_id, Label.key_id)
        
        # Iteratively contruct sparse matrix
        for cid, kid, val in q.all():
            if cid not in row_index:
                row_index[cid] = len(row_index)
            if kid not in col_index:
                col_index[kid] = len(col_index)
            X[row_index[cid], col_index[kid]] = val

        # Return as CSR sparse matrix
        return X.tocsr()

    # TODO
    def add(self, candidates, annotation_generator, session, annotation_key_set_name):
        raise NotImplementedError()

    # TODO
    def remove(self, candidates, key_name, session, annotation_key_set_name):
        raise NotImplementedError()

    # TODO
    def update(self, candidates, annotation_generator, session, annotation_key_set_name):
        raise NotImplementedError()



class LabelFunctionAnnotator(CandidateAnnotator):
    """Apply labeling functions to the candidates, generating Label annotations"""
    def __init__(self):
        super(LabelFunctionAnnotator, self).__init__(annotation=Label)

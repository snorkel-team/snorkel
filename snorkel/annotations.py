import numpy as np
from pandas import DataFrame
from collections import defaultdict
import scipy.sparse as sparse
from .models import Label, Feature, AnnotationKey, AnnotationKeySet
from types import GeneratorType


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
        # If annotarion_key_set refers to an existing AnnotationKeySet, then use this, and only create annotations
        # which fall within this key_set; otherwise we create a new key set
        key_set = session.query(AnnotationKeySet).filter(AnnotationKeySet.name == annotation_key_set_name).first()
        if key_set is not None:
            new  = False
            keys = dict([(k.name, k) for k in key_set.keys])
            print "Using existing key set with %s keys" % len(keys)
        else:
            new     = True
            key_set = AnnotationKeySet(name=annotation_key_set_name)
            session.add(key_set)
            session.commit()
            keys = {}
            print "Creating new key set..."

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
                        if key_name in keys:
                            key = keys[key_name]
                        elif new:
                            key = AnnotationKey(name=key_name)
                            session.add(key)
                            key_set.append(key)
                            session.commit()
                            keys[key_name] = key
                        else:
                            continue
                        session.add(self.annotation(candidate=candidate, key=key, value=value))
        session.commit()
    
    # TODO
    def add(self, candidates, annotation_generator, session, annotation_key_set_name):
        raise NotImplementedError()

    # TODO
    def remove(self, candidates, key_name, session, annotation_key_set_name):
        raise NotImplementedError()

    # TODO
    def update(self, candidates, annotation_generator, session, annotation_key_set_name):
        raise NotImplementedError()

    # TODO: Enforce proper ordering of candidates (rows) in constructing matrices!
    # TODO: Proper key order...
    def load(self, candidates, session, annotation_key_set_name):
        raise NotImplementedError()


class LabelFunctionAnnotator(CandidateAnnotator):
    """Apply labeling functions to the candidates, generating Label annotations"""
    def __init__(self):
        super(LabelFunctionAnnotator, self).__init__(annotation=Label)

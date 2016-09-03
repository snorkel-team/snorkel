from pandas import DataFrame, Series
import scipy.sparse as sparse
from .utils import matrix_conflicts, matrix_coverage, matrix_overlaps
from .models import Label, Feature, AnnotationKey, AnnotationKeySet, Candidate, CandidateSet, Span
from .utils import get_ORM_instance, ProgressBar
from .features import get_span_feats


class csr_AnnotationMatrix(sparse.csr_matrix):
    """
    An extension of the scipy.sparse.csr_matrix class for holding sparse annotation matrices
    and related helper methods.
    """
    def __init__(self, arg1, **kwargs):
        self.candidate_set = kwargs.pop('candidate_set', None)
        self.key_set       = kwargs.pop('key_set', None)

        # Note that scipy relies on the first three letters of the class to define matrix type...
        super(csr_AnnotationMatrix, self).__init__(arg1, **kwargs)

    def get_candidate(self, i):
        """Return the Candidate object corresponding to row i"""
        return self.candidate_set.order_by(Candidate.id)[i]

    def get_key(self, j):
        """Return the AnnotationKey object corresponding to column j"""
        raise NotImplementedError()

    def stats(self):
        """Return summary stats about the annotations"""
        raise NotImplementedError()

    def get_key_stats(self, weights=None):
        """Return a data frame of per-annotation-key stats"""
        raise NotImplementedError()


class csr_LabelMatrix(csr_AnnotationMatrix):
    def stats(self):
        """Returns a pandas DataFrame with the LFs and various per-LF statistics"""
        # Default LF stats
        d = {
            'j'         : range(self.shape[1]),
            'coverage'  : Series(data=matrix_coverage(self), index=self.lf_names),
            'overlaps'  : Series(data=matrix_overlaps(self), index=self.lf_names),
            'conflicts' : Series(data=matrix_conflicts(self), index=self.lf_names)
        }

        return DataFrame(data=d, index=self.lf_names)


class CandidateAnnotator(object):
    """
    Abstract class for annotating candidates, saving these annotations to DB, and then reloading them as
    sparse matrices; generic operation which covers Annotation subclasses:
        * Feature
        * Label
    E.g. for features, LF labels, human annotator labels, etc.
    """
    def __init__(self, annotation, annotation_matrix=csr_AnnotationMatrix):
        self.annotation = annotation
        if not issubclass(annotation_matrix, csr_AnnotationMatrix):
            raise ValueError('annotation_matrix must be a subclass of csr_AnnotationMatrix')
        self.annotation_matrix = annotation_matrix
    
    def _to_annotation_generator(self, fns):
        """"
        Generic method which takes a set of functions, and returns a generator that yields
        function.__name__, function result pairs.
        """
        def fn_gen(c):
            for f in fns:
                yield f.__name__, f(c)
        return fn_gen

    def _create_annotations(self, session, candidate_set, key_set, f, key_set_mutable):
        """
        Generates annotations for candidates in a candidate set, and persists these to the database.
        
        :param candidate_set: A CandidateSet instance

        :param key_set: An AnnotationKeySet instance

        :param f: Can be either:

            * A function which maps a candidate to a generator key_name, value pairs.  Ex: A feature generator

            * A list of functions, each of which maps from candidates to values; by default, the key_name
                is the function.__name__.  Ex: A list of labeling functions

        :param key_set_mutable: If True, annotations with keys not in the given key set will be added, and the
        key set will be expanded; if False, these annotations will be considered out-of-domain (OOD) and discarded.
        """
        annotation_generator = self._to_annotation_generator(f) if hasattr(f, '__iter__') else f
        seen_key_names = set()
        pb = ProgressBar(len(candidate_set))
        for i, candidate in enumerate(candidate_set):
            pb.bar(i)
            seen_key_names.clear()
            for key_name, value in annotation_generator(candidate):
                if key_name not in seen_key_names and value != 0:
                    seen_key_names.add(key_name)
                    
                    # If the annotation is in the key set already, use this AnnotationKey
                    if key_name in key_set.keys:
                        key = key_set.keys[key_name]

                    # Else, only proceed if key set is mutable, in which case create new AnnotationKey
                    elif key_set_mutable:
                        key = AnnotationKey(name=key_name)
                        session.add(key)
                        key_set.append(key)
                        session.commit()
                    else:
                        continue
                    session.add(self.annotation(candidate=candidate, key=key, value=value))
        pb.close()
        session.commit()
    
    def create(self, session, candidate_set, f, new_key_set=None, existing_key_set=None):
        """
        Generates annotations for candidates in a candidate set, and persists these to the database,
        as well as returning a sparse matrix representation.
        
        :param candidate_set: Can either be a CandidateSet instance or the name of one
        
        :param f: Can be either:

            * A function which maps a candidate to a generator key_name, value pairs.  Ex: A feature generator

            * A list of functions, each of which maps from candidates to values; by default, the key_name
                is the function.__name__.  Ex: A list of labeling functions

        :param new_key_set: An AnnotationKeySet instance or name of a new one to create; if provided, a new
        key set will be created and populated
        
        :param existing_key_set: An AnnotationKeySet instance or name of an existing one; if provided, an
        existing key set will be used to define the space of annotation keys; any annotations with keys not in
        this set will simply be discarded.
        """
        candidate_set = get_ORM_instance(CandidateSet, session, candidate_set)
        if new_key_set is None and existing_key_set is None:
            raise ValueError("Must provide a new or existing AnnotationKeySet or annotation key set name.")
        elif new_key_set is not None:
            key_set = AnnotationKeySet(name=new_key_set)
            session.add(key_set)
            session.commit()
            mutable = True
        else:
            key_set = get_ORM_instance(AnnotationKeySet, session, existing_key_set)
            mutable = False
        self._create_annotations(session, candidate_set, key_set, f, key_set_mutable=mutable)
        return self.load(session, candidate_set, key_set)
    
    def update(self, session, candidate_set, key_set, f):
        """
        Generates annotations for candidates in a candidate set and *adds* them to an existing annotation set,
        also adding the respective keys to the key set; returns a sparse matrix representation of the full
        candidate x annotation_key set.

        :param candidate_set: Can either be a CandidateSet instance or the name of one

        :param key_set: Can either be an AnnotationKeySet instance or the name of one
        
        :param f: Can be either:

            * A function which maps a candidate to a generator key_name, value pairs.  Ex: A feature generator

            * A list of functions, each of which maps from candidates to values; by default, the key_name
                is the function.__name__.  Ex: A list of labeling functions
        """
        candidate_set = get_ORM_instance(CandidateSet, session, candidate_set)
        key_set       = get_ORM_instance(AnnotationKeySet, session, key_set)
        self._create_annotations(session, candidate_set, key_set, f, key_set_mutable=True)
        return self.load(session, candidate_set, key_set)

    def load(self, session, candidate_set, key_set):
        """
        Returns the annotations corresponding to a CandidateSet with N members and an AnnotationKeySet with M
        distinct keys as an N x M CSR sparse matrix.
        """
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

        # Return as an AnnotationMatrix
        return self.annotation_matrix(X, candidate_set=candidate_set, key_set=key_set)


class CandidateLabeler(CandidateAnnotator):
    """Apply labeling functions to the candidates, generating Label annotations"""
    def __init__(self):
        super(CandidateLabeler, self).__init__(Label, annotation_matrix=csr_LabelMatrix)

        
class CandidateFeaturizer(CandidateAnnotator):
    """Apply feature generators to the candidates, generating Feature annotations"""
    def __init__(self):
        super(CandidateFeaturizer, self).__init__(annotation=Feature)

    def create(self, session, candidate_set, new_feature_set=None, existing_feature_set=None):
        """
        Generates featuress for candidates in a candidate set, and persists these to the database,
        as well as returning a sparse matrix representation.  Uses currently supported defaults
        based on Candidate arity and constituent argument Context type.
        
        :param candidate_set: Can either be a CandidateSet instance or the name of one
        
        :param new_feature_set: An AnnotationKeySet instance or name of a new one to create; if provided, a new
        key set will be created and populated
        
        :param existing_feature_set: An AnnotationKeySet instance or name of an existing one; if provided, an
        existing key set will be used to define the space of annotation keys; any annotations with keys not in
        this set will simply be discarded.
        """
        # Identify the arity and argument type of the candidates, assuming it is homogenous
        args = candidate_set.candidates[0].get_arguments()
        if isinstance(args[0], Span):
            f = get_span_feats
        else:
            raise NotImplementedError("CandidateFeaturizer currently handles only Span-type candidates.")
        return super(CandidateFeaturizer, self).create(session, candidate_set, f, new_key_set=new_feature_set, \
            existing_key_set=existing_feature_set)

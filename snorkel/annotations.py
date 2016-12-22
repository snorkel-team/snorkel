from pandas import DataFrame, Series
import scipy.sparse as sparse
from sqlalchemy.sql import bindparam, func, select
from .utils import (
    matrix_accuracy,
    matrix_conflicts,
    matrix_coverage,
    matrix_overlaps,
    matrix_tp,
    matrix_fp,
    matrix_fn,
    matrix_tn
)
from .models import AnnotatorLabel, AnnotatorLabelKey, Label, LabelKey, Feature, FeatureKey, Candidate
from .utils import ProgressBar
from .features import get_span_feats
from sqlalchemy.orm.session import object_session


class csr_AnnotationMatrix(sparse.csr_matrix):
    """
    An extension of the scipy.sparse.csr_matrix class for holding sparse annotation matrices
    and related helper methods.
    """
    def __init__(self, arg1, **kwargs):
        # Note: Currently these need to return None if unset, otherwise matrix copy operations break...
        self.candidate_index    = kwargs.pop('candidate_index', None)
        self.row_index          = kwargs.pop('row_index', None)
        self.annotation_key_cls = kwargs.pop('annotation_key_cls', None)
        self.key_index          = kwargs.pop('key_index', None)
        self.col_index          = kwargs.pop('col_index', None)

        # Note that scipy relies on the first three letters of the class to define matrix type...
        super(csr_AnnotationMatrix, self).__init__(arg1, **kwargs)

    def get_candidate(self, session, i):
        """Return the Candidate object corresponding to row i"""
        return session.query(Candidate).filter(Candidate.id == self.row_index[i]).one()
    
    def get_row_index(self, candidate):
        """Return the row index of the Candidate"""
        return self.candidate_index[candidate.id]

    def get_key(self, session, j):
        """Return the AnnotationKey object corresponding to column j"""
        return session.query(self.annotation_key_cls)\
                .filter(self.annotation_key_cls.id == self.col_index[j]).one()

    def get_col_index(self, key):
        """Return the cow index of the AnnotationKey"""
        return self.key_index[key.id]

    def stats(self):
        """Return summary stats about the annotations"""
        raise NotImplementedError()


class csr_LabelMatrix(csr_AnnotationMatrix):

    def lf_stats(self, session, labels=None, est_accs=None):
        """Returns a pandas DataFrame with the LFs and various per-LF statistics"""
        lf_names = [self.get_key(session, j).name for j in range(self.shape[1])]

        # Default LF stats
        col_names = ['j', 'coverage', 'overlaps', 'conflicts']
        d = {
            'j'         : range(self.shape[1]),
            'coverage'  : Series(data=matrix_coverage(self), index=lf_names),
            'overlaps'  : Series(data=matrix_overlaps(self), index=lf_names),
            'conflicts' : Series(data=matrix_conflicts(self), index=lf_names)
        }
        if labels is not None:
            col_names.extend(['accuracy', 'tp', 'fp', 'fn', 'tn'])
            d['accuracy'] = Series(data=matrix_accuracy(self, labels), index=lf_names)
            d['tp']       = Series(data=matrix_tp(self, labels), index=lf_names)
            d['fp']       = Series(data=matrix_fp(self, labels), index=lf_names)
            d['fn']       = Series(data=matrix_fn(self, labels), index=lf_names)
            d['tn']       = Series(data=matrix_tn(self, labels), index=lf_names)

        if est_accs is not None:
            col_names.extend(['Learned Acc.'])
            d['Learned Acc.'] = Series(data=est_accs, index=lf_names)
        return DataFrame(data=d, index=lf_names)[col_names]


class AnnotationManager(object):
    """
    Abstract class for annotating candidates, saving these annotations to DB, and then reloading them as
    sparse matrices; generic operation which covers Annotation subclasses:
        * Feature
        * Label
    E.g. for features, LF labels, human annotator labels, etc.
    """
    def __init__(self, annotation_cls, annotation_key_cls, candidate_cls, matrix_cls=csr_AnnotationMatrix, default_f=None):
        self.annotation_cls     = annotation_cls
        self.annotation_key_cls = annotation_key_cls
        self.candidate_cls      = candidate_cls
        if not issubclass(matrix_cls, csr_AnnotationMatrix):
            raise ValueError('matrix_cls must be a subclass of csr_AnnotationMatrix')
        self.matrix_cls = matrix_cls
        self.default_f = default_f
    
    # TODO: Delete / rename...
    def create(self, session, f=None, split=None, key_group=None):
        """
        Generates annotations for candidates in a candidate set, and persists these to the database,
        as well as returning a sparse matrix representation.

        :param session: SnorkelSession for the database
        
        :param split: The split of the candidate set to use; if None, defaults to all candidates

        :param key_group: The group to add the new keys to
        
        :param f: Can be either:

            * A function which maps a candidate to a generator key_name, value pairs.  Ex: A feature generator

            * A list of functions, each of which maps from candidates to values; by default, the key_name
                is the function.__name__.  Ex: A list of labeling functions
        """
        return self.update(session, True, f=f, split=split, key_group=key_group)
    
    def update(self, session, expand_key_group, f=None, split=None, key_group=None):
        """See create()"""
        # Prepares arguments
        candidates_query = session.query(self.candidate_cls)
        if split is not None:
            candidates_query = candidates_query.filter(self.candidate_cls.split == split)
        if f is None:
            f = self.default_f

        # Prepares helpers
        annotation_generator = _to_annotation_generator(f) if hasattr(f, '__iter__') else f
        pb = ProgressBar(candidates_query.count())

        # Prepares queries
        key_select_query = select([self.annotation_key_cls.id]).where(self.annotation_key_cls.name == bindparam('name'))
        key_insert_query = self.annotation_key_cls.__table__.insert()

        anno_update_query = self.annotation_cls.__table__.update()
        anno_update_query = anno_update_query.where(self.annotation_cls.candidate_id == bindparam('cid'))
        anno_update_query = anno_update_query.where(self.annotation_cls.key_id == bindparam('kid'))
        anno_update_query = anno_update_query.values(value=bindparam('value'))

        anno_insert_query = self.annotation_cls.__table__.insert()

        # Generates annotations for CandidateSet
        for i, candidate in enumerate(candidates_query.all()):
            pb.bar(i)
            for key_name, value in annotation_generator(candidate):

                # Check if the AnnotationKey already exists, and gets its id
                key_id = session.execute(key_select_query, {'name': key_name}).first()
                if key_id is not None:
                    key_id = key_id[0]

                # If expand_key_group is True, then we will always insert or update the Annotation
                if expand_key_group:

                    # If key_name does not exist in the database already, creates a new record
                    if key_id is None:
                        key_id = session.execute(key_insert_query, {'name': key_name}).inserted_primary_key[0]

                    # Updates the annotation value
                    res = session.execute(anno_update_query, {'cid': candidate.id, 'kid': key_id, 'value': value})
                    if res.rowcount == 0 and value != 0:
                        session.execute(anno_insert_query, {'candidate_id': candidate.id, 'key_id': key_id, 'value': value})

                # Else, if the key already exists in the database, we just update the annotation value
                elif key_id is not None:
                    res = session.execute(anno_update_query, {'cid': candidate.id, 'kid': key_id, 'value': value})
                    if res.rowcount == 0 and value != 0:
                        session.execute(anno_insert_query, {'candidate_id': candidate.id, 'key_id': key_id, 'value': value})
        pb.close()
        session.commit()

        print "Loading sparse %s matrix..." % self.annotation_cls.__name__
        return self.load(session, split, key_group)

    def load(self, session, split=None, key_group=None, key_names=None):
        """
        Returns the annotations corresponding to a split of candidates with N members
        and an AnnotationKey group with M distinct keys as an N x M CSR sparse matrix.
        """
        candidates_query = session.query(self.candidate_cls.id)
        if split is not None:
            candidates_query = candidates_query.filter(self.candidate_cls.split == split)
        candidates_query = candidates_query.order_by(self.candidate_cls.id).yield_per(1000)

        keys_query = session.query(self.annotation_key_cls.id)
        if key_group is not None:
            keys_query = keys_query.filter(self.annotation_key_cls.group == key_group)
        if key_names is not None:
            keys_query = keys_query.filter(self.annotation_key_cls.name.in_(frozenset(key_names)))
        keys_query = keys_query.order_by(self.annotation_key_cls.id).yield_per(1000)

        # Create sparse matrix in LIL format for incremental construction
        X = sparse.lil_matrix((candidates_query.count(), keys_query.count()))

        # First, we query to construct the row index map
        cid_to_row = {}
        row_to_cid = {}
        for cid, in candidates_query.all():
            if cid not in cid_to_row:
                j = len(cid_to_row)

                # Create both mappings
                cid_to_row[cid] = j
                row_to_cid[j]   = cid

        # Second, we query to construct the column index map
        kid_to_col = {}
        col_to_kid = {}
        for kid, in keys_query.all():
            if kid not in kid_to_col:
                j = len(kid_to_col)

                # Create both mappings
                kid_to_col[kid] = j
                col_to_kid[j]   = kid

        # NOTE: This is much faster as it allows us to skip the above join (which for some reason is
        # unreasonably slow) by relying on our symbol tables from above; however this will get slower with
        # The total number of annotations in DB which is weird behavior...
        q = session.query(self.annotation_cls.candidate_id, self.annotation_cls.key_id, self.annotation_cls.value)
        q = q.order_by(self.annotation_cls.candidate_id)
        
        # Iteratively construct row index and output sparse matrix
        for cid, kid, val in q.all():
            if cid in cid_to_row and kid in kid_to_col:
                X[cid_to_row[cid], kid_to_col[kid]] = val

        # Return as an AnnotationMatrix
        return self.matrix_cls(X, candidate_index=cid_to_row, row_index=row_to_cid,\
                annotation_key_cls=self.annotation_key_cls, key_index=kid_to_col, col_index=col_to_kid)


class AnnotatorLabelManager(AnnotationManager):
    """Manager for human annotated labels"""
    def __init__(self, candidate_cls):
        super(AnnotatorLabelManager, self).__init__(AnnotatorLabel, AnnotatorLabelKey, candidate_cls, matrix_cls=csr_LabelMatrix)
        
    def load(self, session, annotator_name, split=None):
        """Load a single key only, i.e. labels from one annotator"""
        # TODO: Should add fnality to load several annotators and reduce via e.g. union, majority vote, etc...
        return super(AnnotatorLabelManager, self).load(session, split=split, key_names=[annotator_name])


class LabelManager(AnnotationManager):
    """Apply labeling functions to the candidates, generating Label annotations"""
    def __init__(self, candidate_cls):
        super(LabelManager, self).__init__(Label, LabelKey, candidate_cls, matrix_cls=csr_LabelMatrix)

        
class FeatureManager(AnnotationManager):
    """Apply feature generators to the candidates, generating Feature annotations"""
    def __init__(self, candidate_cls):
        super(FeatureManager, self).__init__(Feature, FeatureKey, candidate_cls, default_f=get_span_feats)


def _to_annotation_generator(fns):
    """"
    Generic method which takes a set of functions, and returns a generator that yields
    function.__name__, function result pairs.
    """
    def fn_gen(c):
        for f in fns:
            yield f.__name__, f(c)
    return fn_gen

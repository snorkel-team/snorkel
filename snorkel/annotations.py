from pandas import DataFrame, Series
import scipy.sparse as sparse
from sqlalchemy.sql import bindparam, func, select
from .udf import UDF
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


class Annotator(UDF):
    """Abstract class for annotating candidates and persisting these annotations to DB"""
    def __init__(self, candidate_subclass, annotation_cls, annotation_key_cls, f, in_queue=None, out_queue=None):
        self.candidate_subclass = candidate_subclass
        self.annotation_cls     = annotation_cls
        self.annotation_key_cls = annotation_key_cls
        self.anno_generator     = _to_annotation_generator(f) if hasattr(f, '__iter__') else f

        # For caching key ids during the reduce step
        self.key_cache  = {}

        super(Annotator, self).__init__(in_queue=in_queue, out_queue=out_queue)
    
    def apply(self, cid, **extra_kwargs):
        """
        Applies a given function to a Candidate, yielding a set of Annotations as key_name, value pairs

        Note: Accepts a candidate _id_ as argument, because of issues with putting Candidate subclasses
        into Queues (can't pickle...)
        """
        seen = set()
        c    = self.session.query(self.candidate_subclass).filter(self.candidate_subclass.id == cid).one()
        for key_name, value in self.anno_generator(c):

            # Note: Make sure no duplicates emitted here!
            if (cid, key_name) not in seen:
                seen.add((cid, key_name))
                yield cid, key_name, value

    def clear(self, cids, create_new_keyset=True, key_group=None, **extra_kwargs):
        """
        Delete all the Annotations associated with the given Candidates (Candidate.ids supplied)
        If create_new_keyset = True, also delete *all* AnnotationKeys of this class,
        or just all AnnotationKeys in key_group if key_group is not None
        """
        # If we are creating a new key set, delete *all* annotations
        qa = self.session.query(self.annotation_cls)
        if not create_new_keyset:
            qa = qa.filter(self.annotation_cls.candidate_id.in_(frozenset(cids)))
        qa.delete(synchronize_session='fetch')

        # If we are creating a new key set, delete all old annoation keys
        if create_new_keyset:
            qk = self.session.query(self.annotation_key_cls)
            if key_group is not None:
                qk = qk.filter(self.annotation_key_cls.group == key_group)
            #qk.delete(synchronize_session='fetch')
            qk.delete()
        self.session.commit()

    def reduce(self, y, create_new_keyset=True, key_group=None, clean_start=False, **extra_kwargs):
        """
        Inserts Annotations into the database.
        For Annotations with unseen AnnotationKeys (in key_group, if not None), either adds these
        AnnotationKeys if create_new_keyset is True, else skips these Annotations.
        """
        cid, key_name, value = y

        ## Prepares queries
        # Key selection & annotation updating only needs to be done if Annotations / AnnotationKeys already exist
        if not clean_start:
            key_select_query = select([self.annotation_key_cls.id])\
                                .where(self.annotation_key_cls.name == bindparam('name'))
            if key_group is not None:
                key_select_query = key_select_query.where(self.annotation_key_cls.group == key_group)

            anno_update_query = self.annotation_cls.__table__.update()
            anno_update_query = anno_update_query.where(self.annotation_cls.candidate_id == bindparam('cid'))
            anno_update_query = anno_update_query.where(self.annotation_cls.key_id == bindparam('kid'))
            anno_update_query = anno_update_query.values(value=bindparam('value'))

        key_insert_query = self.annotation_key_cls.__table__.insert()

        anno_insert_query = self.annotation_cls.__table__.insert()

        # Check if the AnnotationKey already exists, and gets its id
        key_id = None
        if key_name in self.key_cache:
            key_id = self.key_cache[key_name]
        else:
            key_args = {'name': key_name, 'group': key_group} if key_group else {'name': key_name}

            # If AnnotationKeys may exist in DB, check there
            if not clean_start:
                key_id = self.session.execute(key_select_query, key_args).first()

            # Key not in cache but exists in DB; add to cache
            if key_id is not None:
                key_id                   = key_id[0]
                self.key_cache[key_name] = key_id
            
            # Key not in cache or DB; add to both if create_new_keyset = True
            elif create_new_keyset:
                key_id   = self.session.execute(key_insert_query, key_args).inserted_primary_key[0]
                self.key_cache[key_name] = key_id

        # If AnnotationKey does not exist and create_new_keyset = False, skip
        if key_id is not None:

            # Updates the Annotation, assuming one might already exist, if try_update = True
            if not clean_start:
                res = self.session.execute(anno_update_query, {'cid': cid, 'kid': key_id, 'value': value})

            # If Annotation does not exist, insert
            if (clean_start or res.rowcount == 0) and value != 0:
                self.session.execute(anno_insert_query, {'candidate_id': cid, 'key_id': key_id, 'value': value})


class LabelAnnotator(Annotator):
    """Apply labeling functions to the candidates, generating Label annotations"""
    def __init__(self, candidate_subclass, f, in_queue=None, out_queue=None):
        super(LabelAnnotator, self).__init__(candidate_subclass, Label, LabelKey, f, in_queue=in_queue, out_queue=out_queue)

        
class FeatureAnnotator(Annotator):
    """Apply feature generators to the candidates, generating Feature annotations"""
    def __init__(self, candidate_subclass, f=get_span_feats, in_queue=None, out_queue=None):
        super(FeatureAnnotator, self).__init__(candidate_subclass, Feature, FeatureKey, f, in_queue=in_queue, out_queue=out_queue)


def load_matrix(matrix_cls, annotation_key_cls, annotation_cls, session, cids, key_group=None, key_names=None):
    """
    Returns the annotations corresponding to a split of candidates with N members
    and an AnnotationKey group with M distinct keys as an N x M CSR sparse matrix.
    """
    keys_query = session.query(annotation_key_cls.id)
    if key_group is not None:
        keys_query = keys_query.filter(annotation_key_cls.group == key_group)
    if key_names is not None:
        keys_query = keys_query.filter(annotation_key_cls.name.in_(frozenset(key_names)))
    keys_query = keys_query.order_by(annotation_key_cls.id).yield_per(1000)

    # Create sparse matrix in LIL format for incremental construction
    X = sparse.lil_matrix((len(cids), keys_query.count()))

    # First, we query to construct the row index map
    cids.sort()
    cid_to_row = {}
    row_to_cid = {}
    for cid in cids:
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
    q = session.query(annotation_cls.candidate_id, annotation_cls.key_id, annotation_cls.value)
    q = q.order_by(annotation_cls.candidate_id)
    
    # Iteratively construct row index and output sparse matrix
    for cid, kid, val in q.all():
        if cid in cid_to_row and kid in kid_to_col:
            X[cid_to_row[cid], kid_to_col[kid]] = val

    # Return as an AnnotationMatrix
    return matrix_cls(X, candidate_index=cid_to_row, row_index=row_to_cid,\
            annotation_key_cls=annotation_key_cls, key_index=kid_to_col, col_index=col_to_kid)


def load_label_matrix(session, cids, key_group=None):
    return load_matrix(csr_LabelMatrix, LabelKey, Label, session, cids, key_group=key_group)


def load_feature_matrix(session, cids, key_group=None):
    return load_matrix(csr_AnnotationMatrix, FeatureKey, Feature, session, cids, key_group=key_group)


def load_annotator_labels(session, cids, annotator_name):
    return load_matrix(csr_LabelMatrix, AnnotatorLabelKey, AnnotatorLabel, session, cids, key_names=[annotator_name])


def _to_annotation_generator(fns):
    """"
    Generic method which takes a set of functions, and returns a generator that yields
    function.__name__, function result pairs.
    """
    def fn_gen(c):
        for f in fns:
            yield f.__name__, f(c)
    return fn_gen

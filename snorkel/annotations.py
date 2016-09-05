from pandas import DataFrame, Series
import scipy.sparse as sparse
from sqlalchemy.sql import bindparam, func, select
from .utils import matrix_conflicts, matrix_coverage, matrix_overlaps
from .models import Label, Feature, AnnotationKey, AnnotationKeySet, Candidate, CandidateSet
from .models.annotation import annotation_key_set_annotation_key_association as assoc_table
from .utils import get_ORM_instance, ProgressBar
from .features import get_span_feats
from sqlalchemy.orm.session import object_session


class csr_AnnotationMatrix(sparse.csr_matrix):
    """
    An extension of the scipy.sparse.csr_matrix class for holding sparse annotation matrices
    and related helper methods.
    """
    def __init__(self, arg1, **kwargs):
        # Note: Currently these need to return None if unset, otherwise matrix copy operations break...
        self.candidate_set   = kwargs.pop('candidate_set', None)
        self.candidate_index = kwargs.pop('candidate_index', None)
        self.row_index       = kwargs.pop('row_index', None)
        self.key_set         = kwargs.pop('key_set', None)
        self.key_index       = kwargs.pop('key_index', None)
        self.col_index       = kwargs.pop('col_index', None)

        # Note that scipy relies on the first three letters of the class to define matrix type...
        super(csr_AnnotationMatrix, self).__init__(arg1, **kwargs)

    def get_candidate(self, i):
        """Return the Candidate object corresponding to row i"""
        return object_session(self.candidate_set).query(Candidate)\
                .filter(Candidate.id == self.row_index[i]).one()
    
    def get_row_index(self, candidate):
        """Return the row index of the Candidate"""
        return self.candidate_index[candidate.id]

    def get_key(self, j):
        """Return the AnnotationKey object corresponding to column j"""
        return object_session(self.key_set).query(AnnotationKey)\
                .filter(AnnotationKey.id == self.col_index[j]).one()

    def get_col_index(self, key):
        """Return the cow index of the AnnotationKey"""
        return self.key_index[key.id]

    def stats(self):
        """Return summary stats about the annotations"""
        raise NotImplementedError()


class csr_LabelMatrix(csr_AnnotationMatrix):
    def lf_stats(self):
        """Returns a pandas DataFrame with the LFs and various per-LF statistics"""
        lf_names = [self.get_key(j).name for j in range(self.shape[1])]

        # Default LF stats
        d = {
            'j'         : range(self.shape[1]),
            'coverage'  : Series(data=matrix_coverage(self), index=lf_names),
            'overlaps'  : Series(data=matrix_overlaps(self), index=lf_names),
            'conflicts' : Series(data=matrix_conflicts(self), index=lf_names)
        }
        return DataFrame(data=d, index=lf_names)


class AnnotationManager(object):
    """
    Abstract class for annotating candidates, saving these annotations to DB, and then reloading them as
    sparse matrices; generic operation which covers Annotation subclasses:
        * Feature
        * Label
    E.g. for features, LF labels, human annotator labels, etc.
    """
    def __init__(self, annotation_cls, matrix_cls=csr_AnnotationMatrix, default_f=None):
        self.annotation_cls = annotation_cls
        if not issubclass(matrix_cls, csr_AnnotationMatrix):
            raise ValueError('matrix_cls must be a subclass of csr_AnnotationMatrix')
        self.matrix_cls = matrix_cls
        self.default_f = default_f
    
    def create(self, session, candidate_set, new_key_set, f=None):
        """
        Generates annotations for candidates in a candidate set, and persists these to the database,
        as well as returning a sparse matrix representation.

        :param session: SnorkelSession for the database
        
        :param candidate_set: Can either be a CandidateSet instance or the name of one

        :param new_key_set: Name of a new AnnotationKeySet to create
        
        :param f: Can be either:

            * A function which maps a candidate to a generator key_name, value pairs.  Ex: A feature generator

            * A list of functions, each of which maps from candidates to values; by default, the key_name
                is the function.__name__.  Ex: A list of labeling functions
        """
        candidate_set = get_ORM_instance(CandidateSet, session, candidate_set)
        existing_key_set = session.query(AnnotationKeySet).filter(AnnotationKeySet.name == new_key_set).first()
        if existing_key_set is not None:
            raise ValueError('AnnotationKeySet with name ' + new_key_set +
                             ' already exists in the database. Please specify a new name.')
        key_set = AnnotationKeySet(name=new_key_set)
        session.add(key_set)
        session.commit()

        return self.update(session, candidate_set, key_set, True, f)
    
    def update(self, session, candidate_set, key_set, expand_key_set, f=None):
        """
        Generates annotations for candidates in a candidate set and *adds* them to an existing annotation set,
        also adding the respective keys to the key set; returns a sparse matrix representation of the full
        candidate x annotation_key set.

        :param session: SnorkelSession for the database

        :param candidate_set: Can either be a CandidateSet instance or the name of one

        :param key_set: Can either be an AnnotationKeySet instance or the name of one

        :param expand_key_set: If True, annotations with keys not in the given key set will be added, and the
        key set will be expanded; if False, these annotations will be considered out-of-domain (OOD) and discarded.
        
        :param f: Can be either:

            * A function which maps a candidate to a generator key_name, value pairs.  Ex: A feature generator

            * A list of functions, each of which maps from candidates to values; by default, the key_name
                is the function.__name__.  Ex: A list of labeling functions
        """
        # Prepares arguments
        candidate_set = get_ORM_instance(CandidateSet, session, candidate_set)
        key_set       = get_ORM_instance(AnnotationKeySet, session, key_set)
        if f is None:
            f = self.default_f

        # Prepares helpers
        annotation_generator = _to_annotation_generator(f) if hasattr(f, '__iter__') else f
        pb = ProgressBar(len(candidate_set))

        # Prepares queries
        key_select_query = select([AnnotationKey.id]).where(AnnotationKey.name == bindparam('name'))

        key_insert_query = AnnotationKey.__table__.insert()

        assoc_select_query = select([func.count()]).select_from(assoc_table)
        assoc_select_query = assoc_select_query.where(assoc_table.c.annotation_key_set_id == bindparam('ksid'))
        assoc_select_query = assoc_select_query.where(assoc_table.c.annotation_key_id == bindparam('kid'))

        assoc_insert_query = assoc_table.insert()

        anno_update_query = self.annotation_cls.__table__.update()
        anno_update_query = anno_update_query.where(self.annotation_cls.candidate_id == bindparam('cid'))
        anno_update_query = anno_update_query.where(self.annotation_cls.key_id == bindparam('kid'))
        anno_update_query = anno_update_query.values(value=bindparam('value'))

        anno_insert_query = self.annotation_cls.__table__.insert()

        # Generates annotations for CandidateSet
        for i, candidate in enumerate(candidate_set):
            pb.bar(i)
            for key_name, value in annotation_generator(candidate):
                # Check if the AnnotationKey already exists, and gets its id
                key_id = session.execute(key_select_query, {'name': key_name}).first()
                if key_id is not None:
                    key_id = key_id[0]

                # If expand_key_set is True, then we will always insert or update the Annotation
                if expand_key_set:

                    # If key_name does not exist in the database already, creates a new record
                    if key_id is None:
                        key_id = session.execute(key_insert_query, {'name': key_name}).inserted_primary_key[0]

                    # Adds the AnnotationKey to the AnnotationKeySet
                    if session.execute(assoc_select_query, {'ksid': key_set.id, 'kid': key_id}).scalar() == 0:
                        session.execute(assoc_insert_query, {'annotation_key_set_id': key_set.id, 'annotation_key_id': key_id})

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
        return self.load(session, candidate_set, key_set)

    def load(self, session, candidate_set, key_set):
        """
        Returns the annotations corresponding to a CandidateSet with N members and an AnnotationKeySet with M
        distinct keys as an N x M CSR sparse matrix.

        :param session: SnorkelSession for the database

        :param candidate_set: Can either be a CandidateSet instance or the name of one

        :param key_set: Can either be an AnnotationKeySet instance or the name of one
        """
        candidate_set = get_ORM_instance(CandidateSet, session, candidate_set)
        key_set       = get_ORM_instance(AnnotationKeySet, session, key_set)

        # Create sparse matrix in LIL format for incremental construction
        X = sparse.lil_matrix((len(candidate_set), len(key_set)))

        # First, we query to construct the row index map
        cid_to_row = {}
        row_to_cid = {}
        q = session.query(Candidate.id).filter(Candidate.sets.contains(candidate_set)).order_by(Candidate.id).yield_per(1000)
        for cid, in q.all():
            if cid not in cid_to_row:
                j = len(cid_to_row)

                # Create both mappings
                cid_to_row[cid] = j
                row_to_cid[j]   = cid

        # Second, we query to construct the column index map
        kid_to_col = {}
        col_to_kid = {}
        q = session.query(AnnotationKey.id).filter(AnnotationKey.sets.contains(key_set)).order_by(AnnotationKey.id).yield_per(1000)
        for kid, in q.all():
            if kid not in kid_to_col:
                j = len(kid_to_col)

                # Create both mappings
                kid_to_col[kid] = j
                col_to_kid[j]   = kid

        # Construct the query
        """
        q = session.query(self.annotation_cls.candidate_id, self.annotation_cls.key_id, self.annotation_cls.value)
        q = q.join(Candidate, AnnotationKey)
        q = q.filter(Candidate.sets.contains(candidate_set)).filter(AnnotationKey.sets.contains(key_set))
        q = q.order_by(self.annotation_cls.candidate_id).yield_per(1000)
        """

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
        return self.matrix_cls(X, candidate_set=candidate_set, candidate_index=cid_to_row, row_index=row_to_cid,
                               key_set=key_set, key_index=kid_to_col, col_index=col_to_kid)


class LabelManager(AnnotationManager):
    """Apply labeling functions to the candidates, generating Label annotations"""
    def __init__(self):
        super(LabelManager, self).__init__(Label, matrix_cls=csr_LabelMatrix)

        
class FeatureManager(AnnotationManager):
    """Apply feature generators to the candidates, generating Feature annotations"""
    def __init__(self):
        super(FeatureManager, self).__init__(Feature, default_f=get_span_feats)


def _to_annotation_generator(fns):
    """"
    Generic method which takes a set of functions, and returns a generator that yields
    function.__name__, function result pairs.
    """
    def fn_gen(c):
        for f in fns:
            yield f.__name__, f(c)
    return fn_gen

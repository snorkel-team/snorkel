import codecs
import subprocess
import tempfile
from collections import namedtuple
from itertools import izip

import numpy as np
import scipy.sparse as sparse
from pandas import DataFrame, Series

from snorkel.annotations import _to_annotation_generator, FeatureAnnotator
from snorkel.models import Candidate
from snorkel.models.meta import *
from snorkel.udf import UDF, UDFRunner
from snorkel.utils import (
    matrix_conflicts,
    matrix_coverage,
    matrix_overlaps,
    matrix_tp,
    matrix_fp,
    matrix_fn,
    matrix_tn
)
from snorkel.utils import remove_files
from .features.features import get_all_feats

# Used to conform to existing annotation key API call
# Note that this anontation matrix class can not be replaced with snorkel one
# since we do not have ORM-backed key objects but rather a simple python list.
_TempKey = namedtuple('TempKey', ['id', 'name'])
class csr_AnnotationMatrix(sparse.csr_matrix):
    """
    An extension of the scipy.sparse.csr_matrix class for holding sparse annotation matrices
    and related helper methods.
    """
    def __init__(self, arg1, **kwargs):
        # # Note: Currently these need to return None if unset, otherwise matrix copy operations break...
        # self.session = SnorkelSession()
        # Map candidate id to row id
        self.candidate_index = kwargs.pop('candidate_index', {})
        # Map row id to candidate id
        self.row_index = kwargs.pop('row_index', [])
        # Map col id to key str
        self.keys = kwargs.pop('keys', [])
        # Map key str to col number
        self.key_index = kwargs.pop('key_index', {})

        # Note that scipy relies on the first three letters of the class to define matrix type...
        super(csr_AnnotationMatrix, self).__init__(arg1, **kwargs)

    def get_candidate(self, session, i):
        """Return the Candidate object corresponding to row i"""
        return session.query(Candidate)\
                .filter(Candidate.id == self.row_index[i]).one()

    def get_row_index(self, candidate):
        """Return the row index of the Candidate"""
        return self.candidate_index[candidate.id]

    def get_key(self, j):
        """Return the AnnotationKey object corresponding to column j"""
        return _TempKey(j, self.keys[j])

    def get_col_index(self, key):
        """Return the cow index of the AnnotationKey"""
        return self.key_index[key.id]

    def stats(self):
        """Return summary stats about the annotations"""
        raise NotImplementedError()

    def lf_stats_legacy(self, gold=None):
        """Returns a pandas DataFrame with the LFs and various per-LF statistics"""
        lf_names = self.keys

        if gold is not None:
            d = {
                'j'         : range(self.shape[1]),
                'coverage'  : Series(data=matrix_coverage(self), index=lf_names),
                'overlaps'  : Series(data=matrix_overlaps(self), index=lf_names),
                'conflicts' : Series(data=matrix_conflicts(self), index=lf_names),
                #'accuracy'  : Series(data=matrix_accuracy(self, gold), index=lf_names),
            }
        else:
            # Default LF stats
            d = {
                'j'         : range(self.shape[1]),
                'coverage'  : Series(data=matrix_coverage(self), index=lf_names),
                'overlaps'  : Series(data=matrix_overlaps(self), index=lf_names),
                'conflicts' : Series(data=matrix_conflicts(self), index=lf_names)
            }


        return DataFrame(data=d, index=lf_names)
    
    def lf_stats(self, session, labels=None, est_accs=None):
        """Returns a pandas DataFrame with the LFs and various per-LF statistics"""
        lf_names = [self.get_key(session, j).name for j in range(self.shape[1])]

        # Default LF stats
        col_names = ['j', 'Coverage', 'Overlaps', 'Conflicts']
        d = {
            'j'         : range(self.shape[1]),
            'Coverage'  : Series(data=matrix_coverage(self), index=lf_names),
            'Overlaps'  : Series(data=matrix_overlaps(self), index=lf_names),
            'Conflicts' : Series(data=matrix_conflicts(self), index=lf_names)
        }
        if labels is not None:
            col_names.extend(['TP', 'FP', 'FN', 'TN', 'Empirical Acc.'])
            ls = np.ravel(labels.todense() if sparse.issparse(labels) else labels)
            tp = matrix_tp(self, ls)
            fp = matrix_fp(self, ls)
            fn = matrix_fn(self, ls)
            tn = matrix_tn(self, ls)
            ac = (tp+tn).astype(float) / (tp+tn+fp+fn)
            d['Empirical Acc.'] = Series(data=ac, index=lf_names)
            d['TP']             = Series(data=tp, index=lf_names)
            d['FP']             = Series(data=fp, index=lf_names)
            d['FN']             = Series(data=fn, index=lf_names)
            d['TN']             = Series(data=tn, index=lf_names)

        if est_accs is not None:
            col_names.append('Learned Acc.')
            d['Learned Acc.'] = Series(data=est_accs, index=lf_names)
        return DataFrame(data=d, index=lf_names)[col_names]

segment_dir = tempfile.gettempdir()
def get_sql_name(text):
    '''
    Create valid SQL identifier as part of a feature storage table name
    '''
    # Normalize identifier
    text = ''.join(c.lower() if c.isalnum() else ' ' for c in text)
    text = '_'.join(text.split())
    return text

def tsv_escape(s):
    if s is None:
        return '\\N'
    # Make sure feature names are still uniquely encoded in ascii
    s = unicode(s)
    s = s.replace('\"', '\\\\"').replace('\t', '\\t')
    if any(c in ',{}' for c in s):
        s = '"' + s + '"'
    return s

def array_tsv_escape(vals):
    return '{' + ','.join(tsv_escape(p) for p in vals) + '}'

def table_exists(con, name):
    cur = con.execute("select exists(select * from information_schema.tables where table_name=%s)", (name,))
    return cur.fetchone()[0]

def copy_postgres(segment_file_blob, table_name, tsv_columns):
    '''
    @var segment_file_blob: e.g. "segment_*.tsv"
    @var table_name: The SQL table name to copy into
    @var tsv_columns: a string listing column names in the segment files
    separated by comma. e.g. "name, age, income"
    '''
    print 'Copying %s to postgres' % table_name
    cmd = ('cat %s | psql %s -U %s -c "COPY %s(%s) '
            'FROM STDIN" --set=ON_ERROR_STOP=true') % \
            (segment_file_blob, DBNAME, DBUSER, table_name, tsv_columns)
    _out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
    print _out

def _segment_filename(table_name, job_id, start = None, end = None):
    suffix = '*'
    if start is not None:
        suffix = str(start)
        if end is not None:
            suffix += '-' + str(end)
    return '%s_%s_%s.tsv' % (table_name, job_id, suffix)

class COOFeatureAnnotator(FeatureAnnotator):
    def __init__(self, f=get_all_feats, **kwargs):
        super(COOFeatureAnnotator, f, **kwargs)
        
class BatchAnnotatorUDF(UDF):
    def __init__(self, f, **kwargs):
        self.anno_generator  = _to_annotation_generator(f) if hasattr(f, '__iter__') else f
        super(BatchAnnotatorUDF, self).__init__(**kwargs)

    def apply(self, batch_range, table_name, split, **kwargs):
        """
        Applies a given function to a range of candidates

        Note: Accepts a id_range as argument, because of issues with putting Candidate subclasses
        into Queues (can't pickle...)
        """
        start, end = batch_range
        file_name  = _segment_filename(table_name, split, self.worker_id)
        segment_path = os.path.join(segment_dir, file_name)
        candidates = self.session.query(Candidate).filter(Candidate.split == split).order_by(Candidate.id).slice(start, end)
        with codecs.open(segment_path, 'a+', encoding='utf-8') as writer:
            for i, candidate in enumerate(candidates):
                # Runs the actual extraction function
                nonzero_kvs = [(k,v) for k,v in self.anno_generator(candidate) if v != 0]
                if nonzero_kvs:
                    keys, values = zip(*nonzero_kvs)
                else:
                    keys = values = []
                row = [unicode(candidate.id), array_tsv_escape(keys), array_tsv_escape(values)]
                writer.write('\t'.join(row) + '\n')
        return
        yield

class BatchAnnotator(UDFRunner):
    """Abstract class for annotating candidates and persisting these annotations to DB"""
    def __init__(self, candidate_type, annotation_type, f, batch_size = 50, **kwargs):
        if isinstance(candidate_type, type): candidate_type = candidate_type.__name__
        self.table_name = get_sql_name(candidate_type) + '_' + annotation_type
        self.key_table_name = self.table_name + '_keys'
        self.annotation_type = annotation_type
        self.batch_size = batch_size
        super(BatchAnnotator, self).__init__(BatchAnnotatorUDF, f=f, **kwargs)
        
    def apply(self, split, key_group=0, replace_key_set=True, update_keys=True, update_values=True, storage=None, ignore_keys=[], **kwargs):
        # Get the cids based on the split, and also the count
        SnorkelSession = new_sessionmaker()
        session   = SnorkelSession()
        # Note: In the current UDFRunner implementation, we load all these into memory and fill a
        # multiprocessing JoinableQueue with them before starting... so might as well load them here and pass in.
        # Also, if we try to pass in a query iterator instead, with AUTOCOMMIT on, we get a TXN error...
        candidates = session.query(Candidate).filter(Candidate.split == split).all()
        cids_count = len(candidates)
        if cids_count == 0:
            raise ValueError('No candidates in current split')

        # Setting up job batches
        chunks    = cids_count / self.batch_size
        batch_range = [(i * self.batch_size, (i + 1) * self.batch_size) for i in xrange(chunks)]
        remainder = cids_count % self.batch_size
        if remainder:
            batch_range.append((chunks * self.batch_size, cids_count))
            
        old_table_name = None
        table_name = self.table_name
        # Run the Annotator
        with snorkel_engine.connect() as con:
            table_already_exists = table_exists(con, table_name)
            if update_values and table_already_exists:
                # Now we extract under a temporary name for merging
                old_table_name = table_name
                table_name += '_updates'
            
            segment_file_blob = os.path.join(segment_dir, _segment_filename(self.table_name, split))
            remove_files(segment_file_blob)
            super(BatchAnnotator, self).apply(batch_range, table_name = self.table_name, split=split, **kwargs)
            
            # Insert and update keys
            if not table_already_exists or old_table_name:
                con.execute('CREATE TABLE %s(candidate_id integer PRIMARY KEY, keys text[] NOT NULL, values real[] NOT NULL)' % table_name)
            copy_postgres(segment_file_blob, table_name, 'candidate_id, keys, values')
            remove_files(segment_file_blob)
        
            # Replace the LIL table with COO if requested
            if storage == 'COO':
                temp_coo_table = table_name + '_COO'
                con.execute('CREATE TABLE %s AS '
                            '(SELECT candidate_id, UNNEST(keys) as key, UNNEST(values) as value from %s)' % (temp_coo_table, table_name))
                con.execute('DROP TABLE %s'%table_name)
                con.execute('ALTER TABLE %s RENAME TO %s' % (temp_coo_table, table_name))
                con.execute('ALTER TABLE %s ADD PRIMARY KEY(candidate_id, key)' % table_name)
                # Update old table
                if old_table_name:
                    con.execute('INSERT INTO %s SELECT * FROM %s ON CONFLICT(candidate_id, key) '
                                'DO UPDATE SET value=EXCLUDED.value'%(old_table_name, table_name))
                    con.execute('DROP TABLE %s' % table_name)
            else:# LIL
                # Update old table
                if old_table_name:
                    con.execute('INSERT INTO %s AS old SELECT * FROM %s ON CONFLICT(candidate_id) '
                                'DO UPDATE SET '
                                'values=old.values || EXCLUDED.values,'
                                'keys=old.keys || EXCLUDED.keys'%(old_table_name, table_name))
                    con.execute('DROP TABLE %s' % table_name)
            
            if old_table_name: table_name = old_table_name
            # Load the matrix
            key_table_name = self.key_table_name
            if key_group:
                key_table_name = self.key_table_name + '_' + get_sql_name(key_group)
                
            return load_annotation_matrix(con, candidates, split, table_name, key_table_name, replace_key_set, storage, update_keys, ignore_keys)

    def clear(self, session, split, replace_key_set = False, **kwargs):
        """
        Deletes the Annotations for the Candidates in the given split.
        If replace_key_set=True, deletes *all* Annotations (of this Annotation sub-class)
        and also deletes all AnnotationKeys (of this sub-class)
        """
        with snorkel_engine.connect() as con:
            if split is None:
                con.execute('DROP TABLE IF EXISTS %s' % self.table_name)
            elif table_exists(con, self.table_name):
                con.execute('DELETE FROM %s WHERE candidate_id IN '
                            '(SELECT id FROM candidate WHERE split=%d)' % (self.table_name, split))
            if replace_key_set:
                con.execute('DROP TABLE IF EXISTS %s' % self.key_table_name)

    def apply_existing(self, split, key_group=0, **kwargs):
        """Alias for apply that emphasizes we are using an existing AnnotatorKey set."""
        return self.apply(split, key_group=key_group, replace_key_set=False, **kwargs)


class BatchFeatureAnnotator(BatchAnnotator):
    def __init__(self, candidate_type, **kwargs):
        super(BatchFeatureAnnotator, self).__init__(candidate_type, annotation_type='feature', f=get_all_feats, **kwargs)
        
class BatchLabelAnnotator(BatchAnnotator):
    def __init__(self, candidate_type, lfs, **kwargs):
        super(BatchLabelAnnotator, self).__init__(candidate_type, annotation_type='label', f=lfs, **kwargs)

def load_annotation_matrix(con, candidates, split, table_name, key_table_name, replace_key_set, storage, update_keys, ignore_keys):
    '''
    Loads a sparse matrix from an annotation table
    '''
    if replace_key_set:
        # Recalculate unique keys for this set of candidates
        con.execute('DROP TABLE IF EXISTS %s' % key_table_name)
    if replace_key_set or not table_exists(con, key_table_name):
        if storage == 'COO':
            con.execute('CREATE TABLE %s AS '
                        '(SELECT DISTINCT key FROM %s)' % (key_table_name, table_name))
        else:
            con.execute('CREATE TABLE %s AS '
                        '(SELECT DISTINCT UNNEST(keys) as key FROM %s)' % (key_table_name, table_name))
        con.execute('ALTER TABLE %s ADD PRIMARY KEY(key)' % key_table_name)
    elif update_keys:
        if storage == 'COO':
            con.execute('INSERT INTO %s SELECT DISTINCT key FROM %s '
                        'ON CONFLICT(key) DO NOTHING' % (key_table_name, table_name))
        else:
            con.execute('INSERT INTO %s SELECT DISTINCT UNNEST(keys) as key FROM %s '
                        'ON CONFLICT(key) DO NOTHING' % (key_table_name, table_name))
                
    # The result should be a list of all feature strings, small enough to hold in memory
    # TODO: store the actual index in table in case row number is unstable between queries
    ignore_keys = set(ignore_keys)
    keys = [row[0] for row in con.execute('SELECT * FROM %s' % key_table_name) if row[0] not in ignore_keys]
    key_index = {key:i for i, key in enumerate(keys)}
    # Create sparse matrix in LIL format for incremental construction
    lil_feat_matrix = sparse.lil_matrix((len(candidates), len(keys)), dtype=np.float32)

    row_index = []
    candidate_index = {}
    # Load annotations from database
    # TODO: move this for-loop computation to database for automatic parallelization,
    # avoid communication overhead etc. Try to avoid the log sorting factor using unnest
    if storage == 'COO':
        print 'key size', len(keys)
        print 'candidate size', len(candidates)
        iterator_sql = 'SELECT candidate_id, key, value FROM %s '
        'WHERE candidate_id IN '
        '(SELECT id FROM candidate WHERE split=%d) '
        'ORDER BY candidate_id' % (table_name, split)
        prev_id = None
        i = -1
        for _, (candidate_id, key, value) in enumerate(con.execute(iterator_sql)):
            # Update candidate index tracker
            if candidate_id != prev_id:
                i += 1
                candidate_index[candidate_id] = i
                row_index.append(candidate_id)
                prev_id = candidate_id
            # Only keep known features
            key_id = key_index.get(key, None)
            if key_id is not None:
                lil_feat_matrix[i, key_id] = value

    else:
        iterator_sql = '''SELECT candidate_id, keys, values FROM %s
                          WHERE candidate_id IN
                          (SELECT id FROM candidate WHERE split=%d)
                          ORDER BY candidate_id''' % (table_name, split)
        for i, (candidate_id, c_keys, values) in enumerate(con.execute(iterator_sql)):
            candidate_index[candidate_id] = i
            row_index.append(candidate_id)
            for key, value in izip(c_keys, values):
                # Only keep known features
                key_id = key_index.get(key, None)
                if key_id is not None:
                    lil_feat_matrix[i, key_id] = value

    return csr_AnnotationMatrix(lil_feat_matrix, candidate_index=candidate_index,
                                    row_index=row_index, keys=keys, key_index=key_index)

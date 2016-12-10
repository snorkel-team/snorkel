from pandas import DataFrame, Series
import scipy.sparse as sparse
from utils import matrix_conflicts, matrix_coverage, matrix_overlaps, matrix_accuracy,\
    remove_files
from models import Label, Feature, AnnotationKey, AnnotationKeySet, Candidate, CandidateSet
from models.meta import *
from utils import get_ORM_instance, ProgressBar
from features.features import get_all_feats
from sqlalchemy.orm.session import object_session
from multiprocessing import Process
import subprocess
import csv
import multiprocessing
from itertools import izip
from collections import namedtuple
import numpy as np
import codecs
from async_utils import run_in_parallel

# Used to conform to existing annotation key API call
_TempKey = namedtuple('TempKey', ['id', 'name'])
class csr_AnnotationMatrix(sparse.csr_matrix):
    """
    An extension of the scipy.sparse.csr_matrix class for holding sparse annotation matrices
    and related helper methods.
    """
    def __init__(self, arg1, **kwargs):
        # Note: Currently these need to return None if unset, otherwise matrix copy operations break...
        self.session = SnorkelSession()
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

    def get_candidate(self, i):
        """Return the Candidate object corresponding to row i"""
        return self.session.query(Candidate)\
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

    def lf_stats(self, gold=None):
        """Returns a pandas DataFrame with the LFs and various per-LF statistics"""
        lf_names = self.keys

        if gold is not None:
            d = {
                'j'         : range(self.shape[1]),
                'coverage'  : Series(data=matrix_coverage(self), index=lf_names),
                'overlaps'  : Series(data=matrix_overlaps(self), index=lf_names),
                'conflicts' : Series(data=matrix_conflicts(self), index=lf_names),
                'accuracy'  : Series(data=matrix_accuracy(self, gold), index=lf_names),
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

def _to_annotation_generator(fns):
    """"
    Generic method which takes a set of functions, and returns a generator that yields
    function.__name__, function result pairs.
    """
    def fn_gen(c):
        for f in fns:
            yield f.__name__, f(c)
    return fn_gen

class Annotator(object):

    def __init__(self, fns):
        self.fns = [(fn.__name__, fn) for fn in fns]

    def __call__(self, arg):
        for fname, fn in self.fns:
            yield fname, fn(arg)
    
    def keys(self):
        return [name for name, _fn in self.fns]

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

def _annotate_worker(start, end, name, table_name, annotator):
    '''
    Writes raw rows via psql, bypassing ORM.
    Pipes extraction results to the STDIN of a psql COPY process.
    Efficient in terms both of memory and CPU utilization.
    '''

    # Create separate engine for each worker to prevent concurrent connection problems
    engine = new_engine()
    WorkerSession = new_session(engine)
    session = WorkerSession()
    
    # Computes and pipe rows to the COPY process
    candidates = session.query(CandidateSet).filter(CandidateSet.name == name).one().candidates
    
    segment_filename = os.path.join(os.environ.get('SNORKELHOME', '/tmp/'), '%s_%d-%d.tsv' % (table_name, start, end))
    with codecs.open(segment_filename, 'w', encoding='utf-8') as writer:
        pb = None if start else ProgressBar(end)
        for i, candidate in enumerate(candidates.order_by(Candidate.id).slice(start, end)):
            if pb: pb.bar(i)
            # Runs the actual extraction function
            keys, values = zip(*list(annotator(candidate)))
            row = [unicode(candidate.id), array_tsv_escape(keys), array_tsv_escape(values)]
            writer.write('\t'.join(row) + '\n')
        if pb: pb.close()

    
def annotate(candidates, parallel=0, keyset=None, lfs=[]):
    '''
    Extracts features for candidates in parallel
    @var candidates: CandidateSet to extract features from
    @var parallel: Number of processes to use for extraction
    @var keyset: Name of the feature set to use, same as the candidate set name used.
    @var lfs: labeling functions used to annotate the current set of candidates
    otherwise use cached feature index 
    '''
    suffix = '_labels' if lfs else '_features'
    table_name = get_sql_name(candidates.name) + suffix 
    key_table_name = (get_sql_name(keyset) + suffix if keyset else table_name) + '_keys'
    with snorkel_engine.connect() as con:
        con.execute('DROP TABLE IF EXISTS %s' % table_name)
        # TODO: make label table dense
        con.execute('CREATE TABLE %s(candidate_id integer, keys text[] NOT NULL, values real[] NOT NULL)' % table_name)

        # Assuming hyper-threaded cpus
        if not parallel: parallel = min(40, multiprocessing.cpu_count() / 2)
        annotator = Annotator(lfs) if lfs else get_all_feats
        segment_file_blob = os.path.join(os.environ.get('SNORKELHOME', '/tmp/'), table_name) + '*.tsv'
        
        # Clear any previous run temp files if any
        remove_files(segment_file_blob)
        copy_args = (candidates.name, table_name, annotator)
        run_in_parallel(_annotate_worker, parallel, len(candidates), copy_args=copy_args)
        
        
        cmd = ('cat %s | psql %s -U %s -c "COPY %s(candidate_id, keys, values) '
               'FROM STDIN" --set=ON_ERROR_STOP=true') % (segment_file_blob, DBNAME, DBUSER, table_name)
        print 'COPYing from postgres'
        _out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, shell=True)
        print _out
        remove_files(segment_file_blob)
        con.execute('ALTER TABLE %s ADD PRIMARY KEY(candidate_id)' % table_name)
        
        if keyset is None:
            # Recalculate unique keys for this set of candidates
            con.execute('DROP TABLE IF EXISTS %s' % key_table_name)
            con.execute('CREATE TABLE %s AS (SELECT DISTINCT UNNEST(keys) as key FROM %s)' % (key_table_name, table_name))
        # The result should be a list of all feature strings, small enough to hold in memory
        # TODO: store the actual index in table in case row number is unstable between queries
        keys = [row[0] for row in con.execute('SELECT * FROM %s' % key_table_name)]
        key_index = {key:i for i, key in enumerate(keys)}
        
        # Create sparse matrix in LIL format for incremental construction
        lil_feat_matrix = sparse.lil_matrix((len(candidates), len(keys)), dtype=np.float32)

        row_index = []
        candidate_index = {}
        # Load annotations from database
        # TODO: move this for-loop computation to database for automatic parallelization,
        # avoid communication overhead etc.
        iterator_sql = 'SELECT candidate_id, keys, values FROM %s ORDER BY candidate_id' % table_name
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

from pandas import DataFrame, Series
import scipy.sparse as sparse
from utils import matrix_conflicts, matrix_coverage, matrix_overlaps, matrix_accuracy
from models import Label, Feature, AnnotationKey, AnnotationKeySet, Candidate, CandidateSet
from models.meta import *
from utils import get_ORM_instance, ProgressBar
from features.features import get_all_feats
from sqlalchemy.orm.session import object_session
from multiprocessing import Process
import subprocess
import csv
import multiprocessing
from models.annotation import FeatureVector
from itertools import izip


class csr_AnnotationMatrix(sparse.csr_matrix):
    """
    An extension of the scipy.sparse.csr_matrix class for holding sparse annotation matrices
    and related helper methods.
    """
    def __init__(self, arg1, **kwargs):
        # Note: Currently these need to return None if unset, otherwise matrix copy operations break...
        self.candidate_set = kwargs.pop('candidate_set', None)
        self.candidate_index = kwargs.pop('candidate_index', None)
        self.row_index = kwargs.pop('row_index', None)
        self.keys = kwargs.pop('keys', None)
        self.key_index = kwargs.pop('key_index', None)
        self.col_index = kwargs.pop('col_index', None)

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
        return self.keys[j]

    def get_col_index(self, key):
        """Return the cow index of the AnnotationKey"""
        return self.key_index[key.id]

    def stats(self):
        """Return summary stats about the annotations"""
        raise NotImplementedError()


class csr_LabelMatrix(csr_AnnotationMatrix):
    def lf_stats(self, gold=None):
        """Returns a pandas DataFrame with the LFs and various per-LF statistics"""
        lf_names = [self.get_key(j).name for j in range(self.shape[1])]

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

class AnnotationGenerator(object):

    def __init__(self, fns):
        self.fns = [(fn.__name__, fn) for fn in fns]

    def __call__(self, arg):
        for fname, fn in self.fns:
            yield fname, fn(arg)

def tsv_escape(s):
    if s is None:
        return '\\N'
    # Make sure feature names are still uniquely encoded in ascii
    s = unicode(s).encode('ascii','replace')
    # TODO: make sure new line and tab characters are properly escaped
    return s.replace('\"', '\\"')

def array_tsv_escape(vals):
    return '{' + ','.join(tsv_escape(p) for p in vals) + '}'

def _get_exec_plan(candidates, parallel):
        # Plan workload for each extraction worker
    num_jobs = len(candidates)
    avg_jobs = 1 + num_jobs / parallel
    worker_args = [(candidates.name, i * avg_jobs, (i + 1) * avg_jobs) for i in xrange(parallel)]
    worker_args[-1] = (candidates.name, (parallel - 1) * avg_jobs, num_jobs)
    return worker_args

def _annotate_worker(name, start, end):
    '''
    Writes raw rows via psql, bypassing ORM.
    Pipes extraction results to the STDIN of a psql COPY process.
    Efficient in terms both of memory and CPU utilization.
    '''
    copy_process = subprocess.Popen([
        'psql', DBNAME, '-U', DBUSER,
        '-c', '\COPY feature_vector(candidate_id, keys, values) FROM STDIN',
        '--set=ON_ERROR_STOP=true'
        ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    # Create separate engine for each worker to prevent concurrent connection problems
    engine = create_engine(connection)
    WorkerSession = sessionmaker(bind=engine)
    session = WorkerSession()
    
    # Computes and pipe rows to the COPY process
    writer = csv.writer(copy_process.stdin, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    candidates = session.query(CandidateSet).filter(CandidateSet.name == name).one().candidates
    for candidate in candidates.slice(start, end):
        # Runs the actual extraction function
        keys, values = zip(*list(get_all_feats(candidate)))
        row = [candidate.id, array_tsv_escape(keys), array_tsv_escape(values)]
        writer.writerow(row)
    _out, err = copy_process.communicate()
#     if _out:
#         print "standard output of subprocess:"
#         print _out
    if err:
        print "Error of the COPY subprocess:"
        print err
    copy_process.stdin.close()
    
def extract_features(candidates, parallel=min(40, multiprocessing.cpu_count() / 2), expand_key_set = True):
    '''
    Extracts features for candidates in parallel
    @var candidates: CandidateSet to extract features from
    @var parallel: Number of processes to use for extraction
    @var expand_key_set: loads new feature index from feature_vector table if True
    otherwise use cached feature index 
    '''
    with snorkel_engine.connect() as con:
        con.execute('DELETE FROM feature_vector')

    worker_args = _get_exec_plan(candidates, parallel)        
    if parallel == 1:
        # Run without subprocess for easier debugging
        _annotate_worker(*worker_args[0])
    else:
        print 'Using', parallel, 'threads'
        # Fan-out workload to child processes
        ps = [Process(target=_annotate_worker, args=arg) for arg in worker_args]
        for p in ps: p.start()
        for p in ps: p.join()
        
    key2ids = _generate_feature_ids(expand_key_set)
    
    # Create sparse matrix in LIL format for incremental construction
    lil_feat_matrix = sparse.lil_matrix((len(candidates), len(key2ids)))

    session = SnorkelSession()
    # TODO: change this to raw sql for more performance if this ORM layer is problematic
    for i, fv in enumerate(session.query(FeatureVector).order_by(FeatureVector.candidate_id).yield_per(1000)):
        key_indices = []
        values = []
        for key, value in izip(fv.keys, fv.values):
            # Only keep known features
            key_index = key2ids.get(key, None)
            if key_index is not None:
                key_indices.append(key_index)
                values.append(value)
        # bulk assignment for row
        lil_feat_matrix[i, key_indices] = values
        
    return sparse.csr_matrix(lil_feat_matrix)
#     unique_keys = snorkel_engine.execute()

def _generate_feature_ids(expand_key_set = True):
    '''
    Get unique feature keys from all feature vectors, a reduction step
    after the parallel extraction
    @var expand_key_set: loads new feature index from feature_vector table if True
    otherwise use cached feature index 
    '''
    # Build feature index, reuse the main session
    with snorkel_engine.connect() as con:
        if expand_key_set:
            # TODO: Do we need to cache this table?
            con.execute('DROP TABLE IF EXISTS features')
            con.execute('CREATE TABLE features AS (SELECT DISTINCT UNNEST(keys) as key FROM feature_vector)')

        # The result should be a list of all feature strings, small enough to hold in memory
        # TODO: store the actual index in table in case row number is unstable between queries
        keys = con.execute('SELECT * FROM features')
        key2ids = {key[0]:i for i, key in enumerate(keys)}
    return key2ids
        

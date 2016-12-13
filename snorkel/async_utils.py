'''
Created on Dec 7, 2016

@author: xiao
'''
from multiprocessing.process import Process
from models.meta import *
import subprocess
   
def run_in_parallel(worker, parallel, num_jobs, copy_args = ()):
    '''
    Schedules the number of jobs between parallel number of workers
    and runs them. The worker must have the first two arguments
    as the start and end index of its tasks.
    @var copy_args: a tuple of the same arg for all subprocesses
    '''
    # In case requested parallelism is too large for the number of jobs resulting
    # in extra workers
    parallel = min(parallel, num_jobs)
    avg = num_jobs / parallel
    res = num_jobs % parallel
    worker_args = []
    start = 0
    # Assign job range for each worker, residual work distributed uniformly
    for _ in xrange(parallel):
        end = start + avg
        if res:
            end += 1
            res -= 1
        worker_args.append((start, end) + copy_args)
        start = end 
    # Fan-out workload to child processes
    ps = [Process(target=worker, args=arg) for arg in worker_args]
    for p in ps: p.start()
    for p in ps: p.join()

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
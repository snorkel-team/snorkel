import math
import numpy as np
import scipy.sparse as sparse
from multiprocessing import Process, Queue

def mp_apply_lfs(lfs, candidates, nprocs):
    '''MP + labeling functions
    http://eli.thegreenplace.net/2012/01/16/python-parallelizing-cpu-bound-tasks-with-multiprocessing/
    '''
    #print "Using {} processes...".format(nprocs)
    
    def worker(idxs, out_queue):
        outdict = {}
        for i in idxs:
            outdict[i] = [lfs[i](c) for c in candidates]
        out_queue.put(outdict)

    out_queue = Queue()
    chunksize = int(math.ceil(len(lfs) / float(nprocs)))
    procs = []

    nums = range(0,len(lfs))
    for i in range(nprocs):
        p = Process(
                target=worker,
                args=(nums[chunksize * i:chunksize * (i + 1)],
                      out_queue))
        procs.append(p)
        p.start()

    # Collect all results 
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_queue.get())

    for p in procs:
        p.join()

    X = sparse.lil_matrix((len(candidates), len(lfs)))
    for j in resultdict:
        for i,v in enumerate(resultdict[j]):
            if v != 0:
                X[i,j] = v
    
    return X.tocsr()
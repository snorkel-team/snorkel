'''
Created on Dec 10, 2016

@author: xiao
'''
from models.meta import new_session, new_engine
from utils import ProgressBar, get_ORM_instance
from models import CandidateSet
from models.context import Document, Corpus
from async_utils import run_in_parallel
from multiprocessing import Pool

def _extract_worker(start, end, corpus_name, candidateset_name, extractor):
    # Create separate engine for each worker to prevent concurrent connection problems
    engine = new_engine()
    WorkerSession = new_session(engine)
    session = WorkerSession()
    corpus = get_ORM_instance(Corpus, session, corpus_name)
    candidate_set = get_ORM_instance(CandidateSet, session, candidateset_name)
    # Run extraction
    pb = None if start else ProgressBar(end-start)
    for i, context in enumerate(corpus.documents.order_by(Document.id).slice(start, end)):
        if pb: pb.bar(i)
        extractor._extract_from_context(context, candidate_set, session)
    if pb: pb.close()
    session.commit()

_extractor = None
_candidate_set = None
_worker_session = None
def _init_extraction_worker():
    '''
    Per process initialization for parsing
    '''
    global _worker_corpus
    global _worker_session
    _worker_engine = new_engine()
    _worker_session = new_session(_worker_engine)()

def _extract_worker_rr(doc_id):
    doc = _worker_session.query(Document).filter(Document.id==doc_id).one_or_none()
    _extractor._extract_from_context(doc, _candidate_set, _worker_session)
    _worker_session.commit()

def parallel_extract(session, extractor, corpus, candidateset_name, parallel, dynamic_schedule=True):
    global _extractor
    global _candidate_set
    # Create a candidate set
    c = get_ORM_instance(CandidateSet, session, candidateset_name)
    if c is None:
        c = CandidateSet(name=candidateset_name)
        session.add(c)
        session.commit()
        
    if dynamic_schedule:
        _extractor = extractor
        _candidate_set = c
        pb = ProgressBar(len(corpus))
        pool = Pool(parallel, initializer=_init_extraction_worker)
        args = [doc.id for doc in corpus.documents]
        #print 'Working on ', fpaths
        for i, _result in enumerate(pool.imap_unordered(_extract_worker_rr, args)):
            pb.bar(i)
        pool.close()
        pool.join()
        pb.close()
    else:
        default_args = (corpus.name, candidateset_name, extractor)
        # Run extraction jobs
        run_in_parallel(_extract_worker, parallel, len(corpus), default_args)
    return get_ORM_instance(CandidateSet, session, candidateset_name)  

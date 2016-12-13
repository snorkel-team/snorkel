'''
Created on Dec 10, 2016

@author: xiao
'''
from models.meta import new_session, new_engine
from utils import ProgressBar, get_ORM_instance
from models import CandidateSet
from models.context import Document, Corpus
from async_utils import run_in_parallel

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

def parallel_extract(session, extractor, corpus, candidateset_name, parallel):
    # Create a candidate set
    c = get_ORM_instance(CandidateSet, session, candidateset_name)
    if c is None:
        c = CandidateSet(name=candidateset_name)
        session.add(c)
        session.commit()
    default_args = (corpus.name, candidateset_name, extractor)
    # Run extraction jobs
    run_in_parallel(_extract_worker, parallel, len(corpus), default_args)
    return get_ORM_instance(CandidateSet, session, candidateset_name)

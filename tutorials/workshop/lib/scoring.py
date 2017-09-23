import math
import numpy as np
import scipy.sparse as sparse
from multiprocessing import Process, Queue
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from snorkel.models import FeatureKey, GoldLabel, Sentence, Span, Candidate
from snorkel.learning.utils import print_scores

def majority_vote(L):
    '''Majority vote'''
    pred = L.sum(axis=1)
    pred[(pred > 0).nonzero()[0]] = 1
    pred[(pred < 0).nonzero()[0]] = 0
    return pred

def majority_vote_score(L, gold_labels):
    
    y_pred = np.ravel(majority_vote(L))
    y_true = gold_labels.todense()
    y_true = [1 if y_true[i] == 1 else 0 for i in range(y_true.shape[0])]
    
    pos,neg = y_true.count(1),float(y_true.count(0))
    print "pos/neg    {:d}:{:d} {:.1f}%/{:.1f}%".format(int(pos), int(neg), pos/(pos+neg)*100, neg/(pos+neg)*100)
    print "precision  {:.2f}".format( 100 * precision_score(y_true, y_pred) )
    print "recall     {:.2f}".format( 100 * recall_score(y_true, y_pred) )
    print "f1         {:.2f}".format( 100 * f1_score(y_true, y_pred) )
    #print "accuracy  {:.2f}".format( 100 * accuracy_score(y_true, y_pred) 

def apply_lfs(session, lf, split=None, cands=None, gold=None):

    assert split != None or cands != None

    labeled = []
    cands = session.query(Candidate).filter(Candidate.split == split).order_by(Candidate.id).all() \
        if not cands else cands
    for i,c in enumerate(cands):
        if lf(c) != 0:
            if gold != None and gold.size != 0:
                labeled.append((c, gold[i,0]))
            else:
                labeled.append(c)
    return labeled

def coverage(session, lf, split=None, cands=None):
    """

    :param session:
    :param lf:
    :param split:
    :param cands:
    :return:
    """

    cands = session.query(Candidate).filter(Candidate.split == split).order_by(Candidate.id).all() \
        if not cands else cands

    hits = apply_lfs(session, lf, cands=cands)

    v = float(len(hits)) / len(cands) * 100

    print "Coverage: {:2.2f}% ({}/{})".format(v, len(hits), len(cands))
    return hits


def score(session, lf, split, gold, unlabled_as_neg=False):

    cands = session.query(Candidate).filter(Candidate.split == split).order_by(Candidate.id).all()

    tp, fp, tn, fn = [], [], [], []
    for i,c in enumerate(cands):
        label = lf(c)
        label = -1 if label == 0 and unlabled_as_neg else label

        if label == -1 and gold[i, 0] == 1:
            fn += [c]
        elif label == -1 and gold[i, 0] == -1:
            tn += [c]
        elif label == 1 and gold[i, 0] == 1:
            tp += [c]
        elif label == 1 and gold[i, 0] == -1:
            fp += [c]

    print_scores(len(tp), len(fp), len(tn), len(fn), title='LF Score')


def error_analysis(session, lf, split, gold, unlabled_as_neg=False):

    cands = session.query(Candidate).filter(Candidate.split == split).order_by(Candidate.id).all()

    tp, fp, tn, fn = [], [], [], []
    for i,c in enumerate(cands):
        label = lf(c)
        label = -1 if label == 0 and unlabled_as_neg else label

        if label == -1 and gold[i, 0] == 1:
            fn += [c]
        elif label == -1 and gold[i, 0] == -1:
            tn += [c]
        elif label == 1 and gold[i, 0] == 1:
            tp += [c]
        elif label == 1 and gold[i, 0] == -1:
            fp += [c]

    print_scores(len(tp), len(fp), len(tn), len(fn), title='LF Score')
    return tp, fp, tn, fn

def print_top_k_features(session, model, F_matrix, top_k=25):
    """
    Print the top k positive and negatively weighted features.

    :param session:
    :param model:
    :param top_k:
    :return:
    """
    ftrs = session.query(FeatureKey).all()
    ftr_idx = {ftr.id: ftr.name for ftr in ftrs}
    print len(ftr_idx)

    w, b = model.get_weights()

    weights = []
    top_k = 50
    for i in range(0, F_matrix.shape[1]):
        idx = F_matrix.col_index[i]
        weights.append([w[i], ftr_idx[idx]])

    for item in sorted(weights)[0:top_k]:
        print item
    print "-" * 20
    for item in sorted(weights)[-top_k+1:]:
        print item





# def mp_apply_lfs(lfs, candidates, nprocs):
#     '''MP + labeling functions
#     http://eli.thegreenplace.net/2012/01/16/python-parallelizing-cpu-bound-tasks-with-multiprocessing/
#     '''
#
#     # print "Using {} processes...".format(nprocs)
#
#     def worker(idxs, out_queue):
#         outdict = {}
#         for i in idxs:
#             outdict[i] = [lfs[i](c) for c in candidates]
#         out_queue.put(outdict)
#
#     out_queue = Queue()
#     chunksize = int(math.ceil(len(lfs) / float(nprocs)))
#     procs = []
#
#     nums = range(0, len(lfs))
#     for i in range(nprocs):
#         p = Process(
#             target=worker,
#             args=(nums[chunksize * i:chunksize * (i + 1)],
#                   out_queue))
#         procs.append(p)
#         p.start()
#
#     # Collect all results
#     resultdict = {}
#     for i in range(nprocs):
#         resultdict.update(out_queue.get())
#
#     for p in procs:
#         p.join()
#
#     X = sparse.lil_matrix((len(candidates), len(lfs)))
#     for j in resultdict:
#         for i, v in enumerate(resultdict[j]):
#             if v != 0:
#                 X[i, j] = v
#
#     return X.tocsr()
from snorkel.learning.constants import *
from numbskull import NumbSkull
from numbskull.inference import FACTORS
from numbskull.numbskulltypes import Weight, Variable, Factor, FactorToVar
import numpy as np
import scipy.sparse as sparse


def generate_label_matrix(m, n, lf_accuracy_weights, class_prior_weight=None, lf_prior_weights=None,
                          lf_propensity_weights=None, lf_class_propensity_weights=None, deps=(), seed=271828):
    # Compilation

    # Weights
    n_weights = 1 if class_prior_weight is not None else 0

    n_weights += n
    for optional_weights in (lf_prior_weights, lf_propensity_weights, lf_class_propensity_weights):
        if optional_weights is not None:
            if len(optional_weights) != n:
                raise ValueError
            else:
                n_weights += n
    n_weights += len(deps)

    weight = np.zeros(n_weights, Weight)
    for i in range(len(weight)):
        weight[i]['isFixed'] = True

    if class_prior_weight is not None:
        weight[0]['initialValue'] = np.float64(class_prior_weight)
        w_off = 1
    else:
        w_off = 0

    for i in range(n):
        weight[w_off + i]['initialValue'] = np.float64(lf_accuracy_weights[i])
    w_off += n

    for optional_weights in (lf_prior_weights, lf_propensity_weights, lf_class_propensity_weights):
        if optional_weights is not None:
            for i in range(n):
                weight[w_off + i]['initialValue'] = np.float64(optional_weights[i])
            w_off += n

    for i, dep in enumerate(deps):
        weight[w_off + i]['initialValue'] = np.float64(dep[3])

    # Variables
    variable = np.zeros(1 + n, Variable)

    variable[0]['isEvidence'] = 0
    variable[0]['initialValue'] = 0
    variable[0]["dataType"] = 0
    variable[0]["cardinality"] = 2

    for i in range(n):
        variable[1 + i]['isEvidence'] = 0
        variable[1 + i]['initialValue'] = 0
        variable[1 + i]["dataType"] = 0
        variable[1 + i]["cardinality"] = 3

    # Factors and FactorToVar
    n_edges = 1 if class_prior_weight is not None else 0
    n_edges += 2 * n
    if lf_prior_weights is not None:
        n_edges += n
    if lf_propensity_weights is not None:
        n_edges += n
    if lf_class_propensity_weights is not None:
        n_edges += 2 * n
    for dep in deps:
        if dep[2] == DEP_SIMILAR or dep[2] == DEP_EXCLUSIVE:
            n_edges += 2
        elif dep[2] == DEP_FIXING or dep[2] == DEP_REINFORCING:
            n_edges += 3
        else:
            raise ValueError()

    factor = np.zeros(n_weights, Factor)
    ftv = np.zeros(n_edges, FactorToVar)

    if class_prior_weight is not None:
        factor[0]["factorFunction"] = FACTORS["DP_GEN_CLASS_PRIOR"]
        factor[0]["weightId"] = 0
        factor[0]["featureValue"] = 1
        factor[0]["arity"] = 1
        factor[0]["ftv_offset"] = 0

        ftv[0]["vid"] = 0

        f_off = 1
        ftv_off = 1
    else:
        f_off = 0
        ftv_off = 0

    for i in range(n):
        factor[f_off + i]["factorFunction"] = FACTORS["DP_GEN_LF_ACCURACY"]
        factor[f_off + i]["weightId"] = f_off + i
        factor[f_off + i]["featureValue"] = 1
        factor[f_off + i]["arity"] = 2
        factor[f_off + i]["ftv_offset"] = ftv_off + 2 * i

        ftv[ftv_off + 2 * i]["vid"] = 0
        ftv[ftv_off + 2 * i + 1]["vid"] = 1 + i
    f_off += n
    ftv_off += 2 * n

    if lf_prior_weights is not None:
        for i in range(n):
            factor[f_off + i]["factorFunction"] = FACTORS["DP_GEN_LF_PRIOR"]
            factor[f_off + i]["weightId"] = f_off + i
            factor[f_off + i]["featureValue"] = 1
            factor[f_off + i]["arity"] = 1
            factor[f_off + i]["ftv_offset"] = ftv_off + i

            ftv[ftv_off + i]["vid"] = 1 + i
        f_off += n
        ftv_off += n

    if lf_propensity_weights is not None:
        for i in range(n):
            factor[f_off + i]["factorFunction"] = FACTORS["DP_GEN_LF_PROPENSITY"]
            factor[f_off + i]["weightId"] = f_off + i
            factor[f_off + i]["featureValue"] = 1
            factor[f_off + i]["arity"] = 1
            factor[f_off + i]["ftv_offset"] = ftv_off + i

            ftv[ftv_off + i]["vid"] = 1 + i
        f_off += n
        ftv_off += n

    if lf_class_propensity_weights is not None:
        for i in range(n):
            factor[f_off + i]["factorFunction"] = FACTORS["DP_GEN_LF_CLASS_PROPENSITY"]
            factor[f_off + i]["weightId"] = f_off + i
            factor[f_off + i]["featureValue"] = 1
            factor[f_off + i]["arity"] = 2
            factor[f_off + i]["ftv_offset"] = ftv_off + 2 * i

            ftv[ftv_off + 2 * i]["vid"] = 0
            ftv[ftv_off + 2 * i + 1]["vid"] = 1 + i
        f_off += n
        ftv_off += 2 * n

    for i, j, dep, _ in deps:
        if dep == DEP_SIMILAR or dep == DEP_EXCLUSIVE:
            factor[f_off]["factorFunction"] = FACTORS["EQUAL"] if dep == DEP_SIMILAR else FACTORS["DP_GEN_DEP_EXCLUSIVE"]

            factor[f_off]["weightId"] = f_off
            factor[f_off]["featureValue"] = 1
            factor[f_off]["arity"] = 2
            factor[f_off]["ftv_offset"] = ftv_off

            ftv[ftv_off]["vid"] = 1 + i
            ftv[ftv_off + 1]["vid"] = 1 + j

            f_off += 1
            ftv_off += 2
        elif dep == DEP_FIXING or dep == DEP_REINFORCING:
            factor[f_off]["factorFunction"] = FACTORS["DP_GEN_DEP_FIXING"] if dep == DEP_FIXING else FACTORS["DP_GEN_DEP_REINFORCING"]

            factor[f_off]["weightId"] = f_off
            factor[f_off]["featureValue"] = 1
            factor[f_off]["arity"] = 3
            factor[f_off]["ftv_offset"] = ftv_off

            ftv[ftv_off]["vid"] = 0
            ftv[ftv_off + 1]["vid"] = 1 + i
            ftv[ftv_off + 2]["vid"] = 1 + j

            f_off += 1
            ftv_off += 3
        else:
            raise ValueError()

    # Domain mask
    domain_mask = np.zeros(1 + n, np.bool)

    print "Weights"
    print weight
    print "Variables"
    print variable
    print "Factors"
    print factor
    print "FactorToVar"
    print ftv

    # Instantiates factor graph
    ns = NumbSkull(n_inference_epoch=100, quiet=True)
    ns.loadFactorGraph(weight, variable, factor, ftv, domain_mask, n_edges)
    fg = ns.getFactorGraph()

    y = np.ndarray((m,), np.int64)
    L = sparse.lil_matrix((m, n))
    for i in range(m):
        fg.burnIn(100, False)
        y[i] = 1 if fg.var_value[0, 0] == 1 else -1
        for j in range(n):
            L[i, j] = fg.var_value[0, 1 + j] - 1

    return y, L

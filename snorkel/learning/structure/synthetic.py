from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from numbskull import NumbSkull
from numbskull.inference import FACTORS
from numbskull.numbskulltypes import Weight, Variable, Factor, FactorToVar
import numpy as np
import random
import scipy.sparse as sparse
from snorkel.learning import GenerativeModel, GenerativeModelWeights


def generate_model(n, dep_density, class_prior=False, lf_propensity=False, lf_prior=False, lf_class_propensity=False,
                   dep_similar=False, dep_reinforcing=False, dep_fixing=False, dep_exclusive=False, force_dep=False):
    weights = GenerativeModelWeights(n)
    for i in range(n):
        weights.lf_accuracy[i] = 1.1 - 0.2 * random.random()

    if class_prior:
        weights.class_prior = random.choice((-1.0, -2.0))

    if lf_propensity:
        for i in range(n):
            weights.lf_propensity[i] = random.choice((-1.0, -2.0))

    if lf_prior:
        for i in range(n):
            weights.lf_prior[i] = random.choice((1.0, -1.0))

    if lf_class_propensity:
        for i in range(n):
            weights.lf_class_propensity[i] = random.choice((1.0, -1.0))

    if dep_similar:
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < dep_density:
                    weights.dep_similar[i, j] = 0.25

    if dep_fixing:
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < dep_density:
                    if random.random() < 0.5:
                        weights.dep_fixing[i, j] = 0.25
                    else:
                        weights.dep_fixing[j, i] = 0.25

    if dep_reinforcing:
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < dep_density:
                    if random.random() < 0.5:
                        weights.dep_reinforcing[i, j] = 0.25
                    else:
                        weights.dep_reinforcing[j, i] = 0.25

    if dep_exclusive:
        for i in range(n):
            for j in range(i+1, n):
                if random.random() < dep_density:
                    weights.dep_exclusive[i, j] = 0.25

    if force_dep and weights.dep_similar.getnnz() == 0 and weights.dep_fixing.getnnz() == 0 \
        and weights.dep_reinforcing.getnnz() == 0 and weights.dep_exclusive.getnnz() == 0:
        return generate_model(n, dep_density, class_prior=class_prior, lf_propensity=lf_propensity, lf_prior=lf_prior,
                              lf_class_propensity=lf_class_propensity, dep_similar=dep_similar, dep_fixing=dep_fixing,
                              dep_reinforcing=dep_reinforcing, dep_exclusive=dep_exclusive, force_dep=True)
    else:
        return weights


def generate_label_matrix(weights, m):
    # Compilation

    # Weights
    n_weights = 1 if weights.class_prior != 0.0 else 0

    n_weights += weights.n

    for optional_name in GenerativeModel.optional_names:
        for i in range(weights.n):
            if getattr(weights, optional_name)[i] != 0.0:
                n_weights += 1

    for dep_name in GenerativeModel.dep_names:
        for i in range(weights.n):
            for j in range(weights.n):
                if getattr(weights, dep_name)[i, j] != 0.0:
                    n_weights += 1

    weight = np.zeros(n_weights, Weight)
    for i in range(len(weight)):
        weight[i]['isFixed'] = True

    if weights.class_prior != 0.0:
        weight[0]['initialValue'] = np.float64(weights.class_prior)
        w_off = 1
    else:
        w_off = 0

    for i in range(weights.n):
        weight[w_off + i]['initialValue'] = np.float64(weights.lf_accuracy[i])
    w_off += weights.n

    for optional_name in GenerativeModel.optional_names:
        for i in range(weights.n):
            if getattr(weights, optional_name)[i] != 0.0:
                weight[w_off]['initialValue'] = np.float64(getattr(weights, optional_name)[i])
                w_off += 1

    for dep_name in GenerativeModel.dep_names:
        for i in range(weights.n):
            for j in range(weights.n):
                if getattr(weights, dep_name)[i, j] != 0.0:
                    weight[w_off]['initialValue'] = np.float64(getattr(weights, dep_name)[i, j])
                    w_off += 1

    # Variables
    variable = np.zeros(1 + weights.n, Variable)

    variable[0]['isEvidence'] = 0
    variable[0]['initialValue'] = 0
    variable[0]["dataType"] = 0
    variable[0]["cardinality"] = 2

    for i in range(weights.n):
        variable[1 + i]['isEvidence'] = 0
        variable[1 + i]['initialValue'] = 0
        variable[1 + i]["dataType"] = 0
        variable[1 + i]["cardinality"] = 3

    # Factors and FactorToVar
    n_edges = 1 if weights.class_prior != 0.0 else 0
    n_edges += 2 * weights.n
    for optional_name in GenerativeModel.optional_names:
        for i in range(weights.n):
            if getattr(weights, optional_name)[i] != 0.0:
                if optional_name == 'lf_prior' or optional_name == 'lf_propensity':
                    n_edges += 1
                elif optional_name == 'lf_class_propensity':
                    n_edges += 2
                else:
                    raise ValueError()
    for dep_name in GenerativeModel.dep_names:
        for i in range(weights.n):
            for j in range(weights.n):
                if getattr(weights, dep_name)[i, j] != 0.0:
                    if dep_name == 'dep_similar' or dep_name == 'dep_exclusive':
                        n_edges += 2
                    elif dep_name == 'dep_fixing' or dep_name == 'dep_reinforcing':
                        n_edges += 3
                    else:
                        raise ValueError()

    factor = np.zeros(n_weights, Factor)
    ftv = np.zeros(n_edges, FactorToVar)

    if weights.class_prior != 0.0:
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

    for i in range(weights.n):
        factor[f_off + i]["factorFunction"] = FACTORS["DP_GEN_LF_ACCURACY"]
        factor[f_off + i]["weightId"] = f_off + i
        factor[f_off + i]["featureValue"] = 1
        factor[f_off + i]["arity"] = 2
        factor[f_off + i]["ftv_offset"] = ftv_off + 2 * i

        ftv[ftv_off + 2 * i]["vid"] = 0
        ftv[ftv_off + 2 * i + 1]["vid"] = 1 + i
    f_off += weights.n
    ftv_off += 2 * weights.n

    for i in range(weights.n):
        if weights.lf_prior[i] != 0.0:
            factor[f_off]["factorFunction"] = FACTORS["DP_GEN_LF_PRIOR"]
            factor[f_off]["weightId"] = f_off
            factor[f_off]["featureValue"] = 1
            factor[f_off]["arity"] = 1
            factor[f_off]["ftv_offset"] = ftv_off

            ftv[ftv_off]["vid"] = 1 + i
            f_off += 1
            ftv_off += 1

    for i in range(weights.n):
        if weights.lf_propensity[i] != 0.0:
            factor[f_off]["factorFunction"] = FACTORS["DP_GEN_LF_PROPENSITY"]
            factor[f_off]["weightId"] = f_off
            factor[f_off]["featureValue"] = 1
            factor[f_off]["arity"] = 1
            factor[f_off]["ftv_offset"] = ftv_off

            ftv[ftv_off]["vid"] = 1 + i
            f_off += 1
            ftv_off += 1

    for i in range(weights.n):
        if weights.lf_class_propensity[i] != 0.0:
            factor[f_off]["factorFunction"] = FACTORS["DP_GEN_LF_CLASS_PROPENSITY"]
            factor[f_off]["weightId"] = f_off
            factor[f_off]["featureValue"] = 1
            factor[f_off]["arity"] = 2
            factor[f_off]["ftv_offset"] = ftv_off

            ftv[ftv_off]["vid"] = 0
            ftv[ftv_off + 1]["vid"] = 1 + i

            f_off += 1
            ftv_off += 2

    for dep_name in GenerativeModel.dep_names:
        for i in range(weights.n):
            for j in range(weights.n):
                if getattr(weights, dep_name)[i, j] != 0.0:
                    if dep_name == 'dep_similar' or dep_name == 'dep_exclusive':
                        factor[f_off]["factorFunction"] = FACTORS["DP_GEN_DEP_SIMILAR"] if dep_name == 'dep_similar' else FACTORS["DP_GEN_DEP_EXCLUSIVE"]
                        factor[f_off]["weightId"] = f_off
                        factor[f_off]["featureValue"] = 1
                        factor[f_off]["arity"] = 2
                        factor[f_off]["ftv_offset"] = ftv_off

                        ftv[ftv_off]["vid"] = 1 + i
                        ftv[ftv_off + 1]["vid"] = 1 + j

                        f_off += 1
                        ftv_off += 2
                    elif dep_name == 'dep_fixing' or dep_name == 'dep_reinforcing':
                        factor[f_off]["factorFunction"] = FACTORS["DP_GEN_DEP_FIXING"] if dep_name == 'dep_fixing' else FACTORS["DP_GEN_DEP_REINFORCING"]

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
    domain_mask = np.zeros(1 + weights.n, np.bool)

    # Instantiates factor graph
    ns = NumbSkull(n_inference_epoch=100, quiet=True)
    ns.loadFactorGraph(weight, variable, factor, ftv, domain_mask, n_edges)
    fg = ns.getFactorGraph()

    y = np.ndarray((m,), np.int64)
    L = sparse.lil_matrix((m, weights.n), dtype=np.int64)
    for i in range(m):
        fg.burnIn(10, False)
        y[i] = 1 if fg.var_value[0, 0] == 0 else -1
        for j in range(weights.n):
            if fg.var_value[0, 1 + j] != 2:
                L[i, j] = 1 if fg.var_value[0, 1 + j] == 0 else -1

    return y, L.tocsr()

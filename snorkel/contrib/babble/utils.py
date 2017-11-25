import csv
import json
import os

import numpy as np

from snorkel.models import Candidate
from snorkel.learning import RandomSearch

from .explanation import Explanation


class ExplanationIO(object):

    def write(self, explanations, fpath):
        with open(fpath, 'w') as tsvfile:
            tsvwriter = csv.writer(tsvfile, delimiter='\t')
            for exp in explanations:
                tsvwriter.writerow([exp.candidate.get_stable_id(), 
                                    exp.label, 
                                    exp.condition.encode('utf-8'), 
                                    exp.semantics])
        fpath = fpath if len(fpath) < 50 else fpath[:20] + '...' + fpath[-30:]
        print("Wrote {} explanations to {}".format(len(explanations), fpath))

    def read(self, fpath):
        with open(fpath, 'r') as tsvfile:
            tsvreader = csv.reader(tsvfile, delimiter='\t')
            num_read = 0
            explanations = []
            for (candidate, label, condition, semantics) in tsvreader:
                explanations.append(
                    Explanation(
                        condition=condition.strip(),
                        label=True if label=='True' else False,
                        candidate=None if candidate == 'None' else candidate,
                        semantics=semantics))
                num_read += 1
        fpath = fpath if len(fpath) < 50 else fpath[:20] + '...' + fpath[-30:]
        print("Read {} explanations from {}".format(num_read, fpath))
        return explanations


def link_explanation_candidates(explanations, candidates):
    """Doc string goes here."""

    target_candidate_ids = set()
    linked = 0
    print("Building list of target candidate ids...")
    for e in explanations:
        if e.candidate is not None and not isinstance(e.candidate, Candidate):
            target_candidate_ids.add(e.candidate)
        elif e.candidate:
            linked += 1
    if linked == len(explanations):
        print("All {} explanations are already linked to candidates.".format(
            len(explanations)))
        return explanations
    else:
        print("Collected {} unique target candidate ids from {} explanations.".format(
            len(target_candidate_ids), len(explanations)))
    if not target_candidate_ids:
        print("No candidate hashes were provided. Skipping linking.")
        return explanations

    candidate_map = {}
    print("Gathering desired candidates...")
    for candidate in candidates:
        if candidate.get_stable_id() in target_candidate_ids:
            candidate_map[candidate.get_stable_id()] = candidate
    if len(candidate_map) < len(target_candidate_ids):
        num_missing = len(target_candidate_ids) - len(candidate_map)
        print("Could not find {} target candidates with the following stable_ids (first 5):".format(
            num_missing))
        num_reported = 0
        for i, c_hash in enumerate(target_candidate_ids):
            if c_hash not in candidate_map:
                print(c_hash)
                num_reported += 1
                if num_reported >= 5:
                    break
        # raise Exception("Could not find {} target candidates.".format(num_missing))

    print("Found {}/{} desired candidates".format(
        len(candidate_map), len(target_candidate_ids)))

    print("Linking explanations to candidates...")
    for e in explanations:
        if not isinstance(e.candidate, Candidate):
            try:
                e.candidate = candidate_map[e.candidate]
                linked += 1
            except KeyError:
                pass
                # raise Exception("Expected candidate with hash {} could not be found.".format(
                #     e.candidate))

    print("Linked {}/{} explanations".format(linked, len(explanations)))

    return explanations


### MODEL TRAINING ROUTINES
def train_model(model_class, X_train, Y_train=None, X_dev=None, Y_dev=None, 
    search_size=1, search_params={}, rand_seed=123, n_threads=1, verbose=False,
    cardinality=None, params_default={}, model_init_params={}, model_name=None,
    save_dir='checkpoints', beta=1.0, eval_batch_size=None, tune_b=False):
        # Add to model init params
        model_init_params['seed'] = rand_seed
        if cardinality:
            model_init_params['cardinality'] = cardinality

        # Run grid search if search space size > 1
        if search_size > 1 and Y_dev is not None:

            # Initialize hyperparameter search
            searcher = RandomSearch(
                model_class,
                search_params,
                X_train,
                Y_train=Y_train,
                n=search_size, 
                model_class_params=model_init_params,
                model_hyperparams=params_default,
                seed=rand_seed,
                save_dir=save_dir
            )
            
            # Run random grid search
            model, run_stats, opt_b = searcher.fit(X_dev, Y_dev, n_threads=n_threads,
                beta=beta, eval_batch_size=eval_batch_size)
            if verbose > 0:
                print(run_stats)
        else:
            if verbose > 0:
                print("Skipping grid search.")
            model = model_class(**model_init_params)

            # Catches disc vs. gen model interface; could be cleaned up...
            try:
                model.train(X_train, Y_train=Y_train, X_dev=X_dev, Y_dev=Y_dev, 
                    **params_default)
            except TypeError:
                model.train(X_train, **params_default)

            opt_b = 0.5
            if tune_b:
                best_score = -1
                for b in [0.1, 0.15, 0.25, 0.5, 0.75, 0.85, 0.9]:
                    run_scores = model.score(X_dev, Y_dev, b=b, beta=beta,
                        batch_size=eval_batch_size)
                    if run_scores[-1] > best_score:
                        best_score = run_scores[-1]
                        opt_b = b

        # Save model + training marginals in main save_dir (vs save_dir/grid_search)
        model.save(model_name=model_name, save_dir=save_dir)
        return model, opt_b


### SCORING FUNCTIONS
def score_binary_marginals(marginals, labels, b=0.5, set_unlabeled_as_neg=True,
    set_null_predictions_as_neg=True, display=False, tol=1e-6):
    """
    Computes the score given marginals (or more general scalar values) for N
    candidates in a binary classification task.

    :param marginals: An N-dim array of float values, either in [0,1] (e.g.
        actual marginal probabilities for the candidates) or general values.
    :param labels: An N-dim array, list. or annotation matrix, with elements in 
        {-1,0,1}
    :param b: Decision threshold: Any marginal p > b + tol => label = 1, and
        p < b - tol => label = -1, otherwise label = 0 (or -1, see below)
    :param set_unlabeled_as_neg: Whether to map 0 labels -> -1, binary setting.
    :param set_null_predictions_as_neg: Whether to remap marginals 
        p \in [b - tol, b + tol] -> -1
    :param display: Print stats before returning
    :param tol: See entry for b.

    Note: This method assumes predictions and labels are properly collated!
    """
    N = marginals.shape[0]

    # Compute coverage, defined as portion of marginals not in [b-tol, b+tol]
    # Note: We do this *before* any remapping of values
    cov = np.where(np.abs(marginals - b) > tol)[0].shape[0] / float(N)

    # Map marginals to {-1, 0, 1}
    predictions = -1 * np.ones(marginals.shape) if set_null_predictions_as_neg \
        else np.zeros(marginals.shape)
    predictions[marginals > b + tol] = 1
    predictions[marginals < b - tol] = -1

    # Either remap or filter out unlabeled (0-valued) test labels
    if set_unlabeled_as_neg:
        labels[np.abs(labels) < tol] = -1
    else:
        predictions = predictions[np.abs(labels) > tol]
        labels = labels[np.abs(labels) > tol]
    
    # Compute and return precision, recall, and F1 score
    pred_for_true = predictions[labels == 1]
    
    # Compute precision- note we don't include null predictions
    true_pos = pred_for_true[pred_for_true == 1].sum()
    pred_pos = predictions[predictions == 1].sum()
    prec = true_pos / float(pred_pos) if pred_pos > 0 else 0.0
    
    # Compute recall
    pos = labels[labels == 1].sum()
    rec = true_pos / float(pos) if pos > 0 else 0.0
    
    # Compute F1 score
    f1 = (2 * prec * rec) / (prec + rec) if prec + rec > 0 else 0.0

    # Optionally display then return
    if display:
        msg = "### Precision: {0:.2f} | Recall: {1:.2f} | F1 Score: {2:.2f}"
        msg += " | Coverage: {3:.2f}"
        print(msg.format(prec, rec, f1, cov))
    return prec, rec, f1, cov


def score_marginals(marginals, labels, set_unlabeled_as_neg=True, b=0.5,
    set_null_predictions_as_neg=True, display=False, tol=1e-6):
    """
    Computes the score given marginals (or more general scalar values) for N
    candidates.

    Calls score_binary_marginals.
    """
    # Convert labels to dense numpy array
    try:
        labels = np.array(labels.todense()).reshape(-1)
    except:
        labels = np.array(labels)
    try:
        marginals = np.array(marginals.todense())
    except:
        marginals = np.array(marginals)

    # Compute accuracy for categorical, or P/R/F1 for binary settings
    return score_binary_marginals(marginals, labels, b=b, 
        set_unlabeled_as_neg=set_unlabeled_as_neg, 
        set_null_predictions_as_neg=set_null_predictions_as_neg,
        display=display, tol=tol)
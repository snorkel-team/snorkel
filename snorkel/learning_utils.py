# Base Python
import cPickle, json, os, sys, warnings
from collections import defaultdict, OrderedDict, namedtuple
import lxml.etree as et

# Scientific modules
import numpy as np
import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", module="matplotlib")
import matplotlib.pyplot as plt
import scipy.sparse as sparse

def precision(pred, gold):
    tp = np.sum((pred == 1) * (gold == 1))
    fp = np.sum((pred == 1) * (gold != 1))
    return 0 if tp == 0 else float(tp) / float(tp + fp)

def recall(pred, gold):
    tp = np.sum((pred == 1) * (gold == 1))
    p  = np.sum(gold == 1)
    return 0 if tp == 0 else float(tp) / float(p)

def f1_score(pred, gold):
    prec = precision(pred, gold)
    rec  = recall(pred, gold)
    return 0 if (prec * rec == 0) else 2 * (prec * rec)/(prec + rec)

def test_scores(pred, gold, return_vals=True, verbose=False):
    """Returns: (precision, recall, f1_score, tp, fp, tn, fn, n_test)"""
    n_t = len(gold)
    if np.sum(gold == 1) + np.sum(gold == -1) != n_t:
        raise ValueError("Gold labels must be in {-1,1}.")
    tp   = np.sum((pred == 1) * (gold == 1))
    fp   = np.sum((pred == 1) * (gold == -1))
    tn   = np.sum((pred == -1) * (gold == -1))
    fn   = np.sum((pred == -1) * (gold == 1))
    prec = tp / float(tp + fp)
    rec  = tp / float(tp + fn)
    f1   = 2 * (prec * rec) / (prec + rec)

    # Print simple report if verbose=True
    if verbose:
        print "=" * 40
        print "Test set size:\t%s" % n_t
        print "-" * 40
        print "Precision:\t%s" % prec
        print "Recall:\t\t%s" % rec
        print "F1 Score:\t%s" % f1
        print "-" * 40
        print "TP: %s | FP: %s | TN: %s | FN: %s" % (tp,fp,tn,fn)
        print "=" * 40
    if return_vals:
        return prec, rec, f1, tp, fp, tn, fn, n_t

def plot_prediction_probability(probs):
    plt.hist(probs, bins=20, normed=False, facecolor='blue')
    plt.xlim((0,1.025))
    plt.xlabel("Probability")
    plt.ylabel("# Predictions")

def plot_accuracy(probs, ground_truth):
    x = 0.1 * np.array(range(11))
    bin_assign = [x[i] for i in np.digitize(probs, x)-1]
    correct = ((2*(probs >= 0.5) - 1) == ground_truth)
    correct_prob = np.array([np.mean(correct[bin_assign == p]) for p in x])
    xc = x[np.isfinite(correct_prob)]
    correct_prob = correct_prob[np.isfinite(correct_prob)]
    plt.plot(x, np.abs(x-0.5) + 0.5, 'b--', xc, correct_prob, 'ro-')
    plt.xlim((0,1))
    plt.ylim((0,1))
    plt.xlabel("Probability")
    plt.ylabel("Accuracy")

def calibration_plots(train_marginals, test_marginals, gold_labels=None):
    """Show classification accuracy and probability histogram plots"""
    n_plots = 3 if gold_labels is not None else 1
    
    # Whole set histogram
    plt.subplot(1,n_plots,1)
    plot_prediction_probability(train_marginals)
    plt.title("(a) # Predictions (training set)")

    if gold_labels is not None:

        # Hold-out histogram
        plt.subplot(1,n_plots,2)
        plot_prediction_probability(test_marginals)
        plt.title("(b) # Predictions (test set)")

        # Classification bucket accuracy
        plt.subplot(1,n_plots,3)
        plot_accuracy(test_marginals, gold_labels)
        plt.title("(c) Accuracy (test set)")
    plt.show()

class DictTable(OrderedDict):
  def set_title(self, heads):
    self.title = heads
  def set_rows(self, n):
    self.rows = n
  def set_cols(self, n):
    self.cols = n
  def _repr_html_(self):
    html = ["<table>"]
    if hasattr(self, 'title'):
      html.append("<tr>")
      html.extend("<td><b>{0}</b></td>".format(t) for t in self.title)
      html.append("</tr>")
    items = self.items()[:self.rows] if hasattr(self, 'rows') else self.items()
    for k, v in items:
      html.append("<tr>")
      html.append("<td>{0}</td>".format(k))
      html.extend("<td>{0}</td>".format(i) for i in v)
      html.append("</tr>")
    html.append("</table>")
    return ''.join(html)

class SideTables:
  def __init__(self, table1, table2):
    self.t1, self.t2 = table1, table2
  def _repr_html_(self):
    t1_html = self.t1._repr_html_()
    t2_html = self.t2._repr_html_()
    t1_html = t1_html[:6] + " style=\"margin-right: 1%;float: left\"" + t1_html[6:] 
    t2_html = t2_html[:6] + " style=\"float: left\"" + t2_html[6:] 
    return t1_html + t2_html


class DDLiteModel:
  def __init__(self, candidates, feats=None, gt_dict=None):
    self.C = candidates
    
  #######################################################
  #################### LF stat comp. ####################
  #######################################################    

  def _cover(self, idxs=None):
    idxs = self.training() if idxs is None else idxs
    return [np.ravel((self.lf_matrix[idxs,:] == lab).sum(1))
            for lab in [1,-1]]

  def coverage(self, cov=None, idxs=None):
    cov = self._cover(idxs) if cov is None else cov    
    return np.mean((cov[0] + cov[1]) > 0)

  def overlap(self, cov=None, idxs=None):    
    cov = self._cover(idxs) if cov is None else cov    
    return np.mean((cov[0] + cov[1]) > 1)

  def conflict(self, cov=None, idxs=None):    
    cov = self._cover(idxs) if cov is None else cov    
    return np.mean(np.multiply(cov[0], cov[1]) > 0)

  def print_lf_stats(self, idxs=None):
    """
    Returns basic summary statistics of the LFs on training set (default) or
    passed idxs
    * Coverage = % of candidates that have at least one label
    * Overlap  = % of candidates labeled by > 1 LFs
    * Conflict = % of candidates with conflicting labels
    """
    cov = self._cover(idxs)
    print "LF stats on training set" if idxs is None else "LF stats on idxs"
    print "Coverage:\t{:.3f}%\nOverlap:\t{:.3f}%\nConflict:\t{:.3f}%".format(
            100. * self.coverage(cov), 
            100. * self.overlap(cov),
            100. * self.conflict(cov))

  def _plot_coverage(self, cov):
    cov_ct = [np.sum(x > 0) for x in cov]
    tot_cov = self.coverage(cov)
    idx, bar_width = np.array([1, -1]), 1
    plt.bar(idx, cov_ct, bar_width, color='b')
    plt.xlim((-1.5, 2.5))
    plt.xlabel("Label type")
    plt.ylabel("# candidates with at least one of label type")
    plt.xticks(idx + bar_width * 0.5, ("Positive", "Negative"))
    return tot_cov * 100.
    
  def _plot_conflict(self, cov):
    x, y = cov
    tot_conf = self.conflict(cov)
    m = np.max([np.max(x), np.max(y)])
    bz = np.linspace(-0.5, m+0.5, num=m+2)
    H, xr, yr = np.histogram2d(x, y, bins=[bz,bz], normed=False)
    plt.imshow(H, interpolation='nearest', origin='low',
               extent=[xr[0], xr[-1], yr[0], yr[-1]])
    cb = plt.colorbar(fraction=0.046, pad=0.04)
    cb.set_label("# candidates")
    plt.xlabel("# negative labels")
    plt.ylabel("# positive labels")
    plt.xticks(range(m+1))
    plt.yticks(range(m+1))
    return tot_conf * 100.

  def plot_lf_stats(self):
    """ Show plots for evaluating LF quality
    Coverage bar plot, overlap histogram, and conflict heat map
    """
    if self.lf_matrix is None:
      raise ValueError("No LFs applied yet")
    n_plots = 2
    cov = self._cover()
    # LF coverage
    plt.subplot(1,n_plots,1)
    tot_cov = self._plot_coverage(cov)
    plt.title("(a) Label balance (training set coverage: {:.2f}%)".format(tot_cov))
    # LF conflict
    plt.subplot(1,n_plots,2)
    tot_conf = self._plot_conflict(cov)
    plt.title("(b) Label heat map (training set conflict: {:.2f}%)".format(tot_conf))
    # Show plots    
    plt.show()

  def _lf_conf(self, lf_idx):
    lf_csc = self.lf_matrix.tocsc()
    other_idx = np.concatenate((range(lf_idx),range(lf_idx+1, self.num_lfs())))
    ts = self.training()
    agree = lf_csc[:, other_idx].multiply(lf_csc[:, lf_idx])
    agree = agree[ts,:]
    return float((np.ravel((agree == -1).sum(1)) > 0).sum()) / len(ts)
    
  def top_conflict_lfs(self, n=10):
    """ Show the LFs with the highest mean conflicts per candidate """
    d = {nm : ["{:.2f}%".format(100.*self._lf_conf(i))]
         for i,nm in enumerate(self.lf_names)}
    tab = DictTable(sorted(d.items(), key=lambda t:t[1], reverse=True))
    tab.set_rows(n)
    tab.set_cols(2)
    tab.set_title(["Labeling function", "Percent candidates where LF has conflict"])
    return tab
    
  def _lf_coverage(self, lf_idx):
    lf_v = np.ravel(self.lf_matrix.tocsc()[self.training(), lf_idx].todense())
    return 1 - np.mean(lf_v == 0)
    
  def lowest_coverage_lfs(self, n=10):
    """ Show the LFs with the highest fraction of abstains """
    d = {nm : ["{:.2f}%".format(100.*self._lf_coverage(i))]
         for i,nm in enumerate(self.lf_names)}
    tab = DictTable(sorted(d.items(), key=lambda t:t[1]))
    tab.set_rows(n)
    tab.set_cols(2)
    tab.set_title(["Labeling function", "Candidate coverage"])
    return tab

  def _lf_acc(self, subset, lf_idx):
    gt = self.gt._gt_vec
    pred = np.ravel(self.lf_matrix.tocsc()[:,lf_idx].todense())
    has_label = np.where(pred != 0)
    has_gt = np.where(gt != 0)
    # Get labels/gt for candidates in dev set, with label, with gt
    gd_idxs = np.intersect1d(has_label, subset)
    gd_idxs = np.intersect1d(has_gt, gd_idxs)
    gt = np.ravel(gt[gd_idxs])
    pred_sub = np.ravel(pred[gd_idxs])
    n_neg = np.sum(pred_sub == -1)
    n_pos = np.sum(pred_sub == 1)
    if np.sum(pred == -1) == 0:
      neg_acc = -1
    elif n_neg == 0:
      neg_acc = 0
    else:
      neg_acc = float(np.sum((pred_sub == -1) * (gt == -1))) / n_neg
    if np.sum(pred == 1) == 0:
      pos_acc = -1
    elif n_pos == 0:
      pos_acc = 0
    else: 
      pos_acc = float(np.sum((pred_sub == 1) * (gt == 1))) / n_pos
    return (pos_acc, n_pos, neg_acc, n_neg)
    
  def _lf_acc_gen(self, lf_idx):
    pos_acc1, n_pos, neg_acc1, n_neg = self._lf_acc(self.dev1(), lf_idx)
    pos_acc2, n_pos2, neg_acc2, n_neg2 = self._lf_acc(self.dev2(), lf_idx)
    pos_acc2, neg_acc2 = max(0, pos_acc2), max(0, neg_acc2)
    return (pos_acc1, n_pos, abs(pos_acc1 - pos_acc2), n_pos2,
            neg_acc1, n_neg, abs(neg_acc1 - neg_acc2), n_neg2)    
    
  def lowest_empirical_accuracy_lfs(self, n=10):
    self.dev_size_warn()
    print "100% accuracy and 0 generalization score are \"perfect\""
    """ Show the LFs with the lowest accuracy compared to ground truth """
    d = {nm : list(self._lf_acc_gen(i)) for i,nm in enumerate(self.lf_names)}
    tab_pos = DictTable(sorted(d.items(), key=lambda t:t[1][0]))
    for k in tab_pos:
      if tab_pos[k][0] < 0:
        del tab_pos[k]
        continue
      tab_pos[k] = ["{:.2f}% (n={})".format(100.*tab_pos[k][0], tab_pos[k][1]),
                    "{:.2f} (n={})".format(tab_pos[k][2], tab_pos[k][3])] 
    tab_pos.set_rows(n)
    tab_pos.set_cols(3)
    tab_pos.set_title(["Labeling function", "Positive accuracy",
                       "Gen. score"])
    
    tab_neg = DictTable(sorted(d.items(), key=lambda t:t[1][4]))
    for k in tab_neg:
      if tab_neg[k][4] < 0:
        del tab_neg[k]
        continue
      tab_neg[k] = ["{:.2f}% (n={})".format(100.*tab_neg[k][4], tab_neg[k][5]),
                    "{:.2f} (n={})".format(tab_neg[k][6], tab_neg[k][7])]
    tab_neg.set_rows(n)
    tab_neg.set_cols(3)
    tab_neg.set_title(["Labeling function", "Negative accuracy",
                       "Gen. score"])
    return SideTables(tab_pos, tab_neg)
    
  def lf_summary_table(self):
    d = {nm : [self._lf_coverage(i), self._lf_conf(i), self._lf_acc_gen(i)]
         for i,nm in enumerate(self.lf_names)}
    for k,v in d.items():
      del d[k]
      pos_k, both_k = (v[2][0] >= 0), (v[2][0] >= 0 and v[2][4] >= 0)
      col, tp, pa, pg, na, ng = ("#ee0b40", "Negative", "N/A", "N/A",
                                 "{:.2f}% (n={})".format(100.*v[2][4], v[2][5]),
                                 "{:.2f} (n={})".format(v[2][6], v[2][7]))
      if pos_k:
        col, tp, na, ng, pa, pg = ("#0099ff", "Positive", "N/A", "N/A",
                                   "{:.2f}% (n={})".format(100.*v[2][0], v[2][1]),
                                   "{:.2f} (n={})".format(v[2][2], v[2][3]))
      if both_k:
        col, tp, pa, pg, na, ng = ("#c700ff", "Both",
                                   "{:.2f}% (n={})".format(100.*v[2][0], v[2][1]),
                                   "{:.2f} (n={})".format(v[2][2], v[2][3]),
                                   "{:.2f}% (n={})".format(100.*v[2][4], v[2][5]),
                                   "{:.2f} (n={})".format(v[2][6], v[2][7]))     
      fancy_k = "<b><font color=\"{}\">{}</font></b>".format(col, k)
      d[fancy_k] = [tp, "{:.2f}%".format(100.*v[0]),
                      "{:.2f}%".format(100.*v[1]), pa, pg, na, ng]
    tab = DictTable(sorted(d.items(), key=lambda t:t[1][0]))
    tab.set_rows(len(self.lf_names))
    tab.set_cols(8)
    tab.set_title(["Labeling<br />function", "Label<br />type",
                   "Candidate<br />coverage", "Candidate<br />conflict", 
                   "Positive<br />accuracy", "Positive<br />gen. score",
                   "Negative<br />accuracy", "Negative<br />gen. score"])
    return tab
    
  def plot_lr_diagnostics(self, w_fit, mu_opt, f1_opt):
    """ Plot validation set performance for logistic regression regularization """
    mu_seq = sorted(w_fit.keys())
    p = np.ravel([w_fit[mu].P for mu in mu_seq])
    r = np.ravel([w_fit[mu].R for mu in mu_seq])
    f1 = np.ravel([w_fit[mu].F1 for mu in mu_seq])
    nnz = np.ravel([np.sum(w_fit[mu].w != 0) for mu in mu_seq])    
    
    fig, ax1 = plt.subplots()
    # Plot spread
    ax1.set_xscale('log', nonposx='clip')    
    ax1.scatter(mu_opt, f1_opt, marker='*', color='purple', s=500,
                zorder=10, label="Maximum F1: mu={}".format(mu_opt))
    ax1.plot(mu_seq, f1, 'o-', color='red', label='F1 score')
    ax1.plot(mu_seq, p, 'o--', color='blue', label='Precision')
    ax1.plot(mu_seq, r, 'o--', color='green', label='Recall')
    ax1.set_xlabel('log(penalty)')
    ax1.set_ylabel('F1 score/Precision/Recall')
    ax1.set_ylim(-0.04, 1.04)
    for t1 in ax1.get_yticklabels():
      t1.set_color('r')
    # Plot nnz
    ax2 = ax1.twinx()
    ax2.plot(mu_seq, nnz, '.:', color='gray', label='Sparsity')
    ax2.set_ylabel('Number of non-zero coefficients')
    ax2.set_ylim(-0.01*np.max(nnz), np.max(nnz)*1.01)
    for t2 in ax2.get_yticklabels():
      t2.set_color('gray')
    # Shrink plot for legend
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0+box1.height*0.1, box1.width, box1.height*0.9])
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0+box2.height*0.1, box2.width, box2.height*0.9])
    plt.title("Validation for logistic regression learning")
    lns1, lbs1 = ax1.get_legend_handles_labels()
    lns2, lbs2 = ax2.get_legend_handles_labels()
    ax1.legend(lns1+lns2, lbs1+lbs2, loc='upper center', bbox_to_anchor=(0.5,-0.05),
               scatterpoints=1, fontsize=10, markerscale=0.5)
    plt.show()

  def _get_all_abstained(self, training=True):
    idxs = self.training() if training else range(self.num_candidates())
    return np.ravel(np.where(np.ravel((self.lf_matrix[idxs,:]).sum(1)) == 0))

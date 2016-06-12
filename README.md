# DeepDive Lite

[![Build Status](https://travis-ci.org/HazyResearch/ddlite.svg?branch=master)](https://travis-ci.org/HazyResearch/ddlite)

#### Sponsored by: 
<img src="figs/darpa.JPG" width="80" height="80" />
#### as part of the [SIMPLEX](http://www.darpa.mil/program/simplifying-complexity-in-scientific-discovery) program under contract number N66001-15-C-4043.

## Motivation
DDLite is intended to be a lightweight but powerful framework for prototyping & developing **structured information extraction applications** for domains in which large labeled training sets are not available or easy to obtain, using the _data programming_ paradigm.

In the data programming approach to developing a machine learning system, the developer focuses on writing a set of _labeling functions_, which create a large but noisy training set. DDLite then learns a generative model of this noise&mdash;learning, essentially, which labeling functions are more accurate than others&mdash;and uses this to train a discriminative classifier.

At a high level, the idea is that developers can focus on writing labeling functions&mdash;which are just (Python) functions that provide a label for some subset of data points&mdash;and not think about algorithms _or_ features!

DDLite is very much a work in progress, but some people have already begun developing applications with it, and initial feedback has been positive... let us know what you think, and how we can improve it, in the [Issues](https://github.com/HazyResearch/ddlite/issues) section!

## Installation / dependencies

<!-- TODO these manual instruction could be abstracted away with a simple launcher script, that takes as input a ipynb and simply opens it after any necessary setup.. -->

First of all, make sure all git submodules have been downloaded.

```bash
git submodule update --init --recursive
git submodule update --recursive
```

DeepDive Lite requires [a few python packages](python-package-requirement.txt) including:

* [nltk](http://www.nltk.org/install.html)
* [lxml](http://lxml.de/installation.html)
* [requests](http://docs.python-requests.org/en/master/user/install/#install)
* [numpy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
* [scipy](http://www.scipy.org/install.html)
* [matplotlib](http://matplotlib.org/users/installing.html)

We provide a simple way to install everything using `virtualenv`:

```bash
# set up a Python virtualenv
virtualenv .virtualenv
source .virtualenv/bin/activate

pip install --requirement python-package-requirement.txt
```

*Note: if you have an issue with the matplotlib install related to the module `freetype`, see [this post](http://stackoverflow.com/questions/20533426/ubuntu-running-pip-install-gives-error-the-following-required-packages-can-no); if you have an issue installing ipython, try [upgrading setuptools](http://stackoverflow.com/questions/35943606/error-on-installing-ipython-for-python-3-sys-platform-darwin-and-platform)*

Alternatively, they could be installed system-wide if `sudo pip` is used instead of `pip` in the last command without the virtualenv setup and activation.

In addition the Stanford CoreNLP parser jars need to be downloaded; this can be done using:
```bash
./install-parser.sh
```

Finally, DeepDive Lite is built specifically with usage in **Jupyter/IPython notebooks** in mind.
The `jupyter` command is installed as part of the above installation steps, so the following command within the virtualenv opens our demo notebook.

```bash
jupyter notebook examples/GeneTaggerExample_Extraction.ipynb
```

## Learning how to use DeepDive Lite
The best way to learn how to use is to open up the demo notebooks in the **examples** folder. **GeneTaggerExample_Extraction.ipynb** walks through the candidate extraction workflow for an entity tagging task. **GeneTaggerExample_Learning.ipynb** picks up where the extraction notebook left off. The learning notebook demonstrates the labeling function iteration workflow and learning methods. For examples of extracting relations, see **GenePhenRelationExample_Extraction.ipynb** and **GenePhenRelationExample_Learning.ipynb**.

## Best practices for labeling function iteration
The flowchart below illustrates the labeling function iteration workflow.

![ddlite workflow](/figs/LFworkflow.png)

First, we generate candidates, a hold out set of candidates (using MindTagger or gold standard labels), and initial set of labeling functions *L<sup>(0)</sup>* for a `CandidateModel`. The next step is to examine the coverage and accuracy of labeling functions using `CandidateModel.plot_lf_stats()`, `CandidateModel.top_conflict_lfs()`, `CandidateModel.lowest_coverage_lfs()`, and `CandidateModel.lowest_empirical_accuracy_lfs()`. If coverage is the primary issue, we write a new candidate function and append it to *L<sup>(0)</sup>* to form *L<sup>(1)</sup>*. If accuracy is the primary issue instead, we form *L<sup>(1)</sup>* by revising an existing labeling function which may be implemented incorrectly. This process continues until we are satisfied with the labeling function set. We then learn a model, and depending on the performance over the hold out set, revaluate our labeling function set as before.

Several parts of this workflow could result in overfitting, so tie your bootstraps with care:

* Generating a sufficiently large and diverse hold out set before iterating on labeling functions is important for accurate evaluation.
* A labeling function with low empirical accuracy on the hold out set could work well on the entire data set. This is not a reason to delete it (unless the implementation is incorrect).
* Metrics obtained by altering learning parameters directly to maximize performance against the hold out set are upwards biased and not representative of general performance.

## Best practices for using DeepDive Lite notebooks
Here are a few practical tips for working with DeepDive Lite:
* Use `autoreload`
* Keep working source code in another file 
* Pickle extractions often and with unique names
* Entire objects (extractions and features) subclassed from `Candidates` can be pickled
* Document past labeling functions either remotely or with the `CandidateModel` log

## Documentation

### **ddlite_parser.py**

#### Class: Sentence

|**Member**|**Notes**|
|:---------|:--------|
|`words`| Tokenized words|
|`lemmas`| Tokenized lemmas|
|`poses`| Tokenized parts-of-speech|
|`dep_parents`| Dependency parent index for each token |
|`dep_labels`|Dependency label for each token|
|`sent_id`| Sentence ID|
|`doc_id`|Document ID|
|`text`| Raw sentence text|
|`token_idxs`| Document character offsets for start of each sentence token |
|`doc_name` | File name of document |

#### Class: SentenceParser

|**Method**|**Notes**|
|:---------|:--------|
|`__init__(tok_whitespace=False)`| Starts CoreNLPServer; if input is pretokenized and space separated, use `tok_whitespace=True` |
|`parse(doc, doc_id=None)` | Parse document into `Sentence`s|

#### Class: HTMLReader

|**Method**|**Notes**|
|:---------|:--------|
|`can_read(f)`||
|`parse(f)`| Returns visible text in HTML file|

#### Class: TextReader

|**Method**|**Notes**|
|:---------|:--------|
|`can_read(f)`||
|`parse(f)`| Returns all text in file|

#### Class: DocParser

|**Method**|**Notes**|
|:---------|:--------|
|`__init__(path, ftreader = TextReader())` | `path` can be single file, a directory, or a glob expression |
|`readDocs()` | Returns docs as parsed by `ftreader` |
|`parseDocSentences()` | Returns `Sentence`s from `SentenceParser` parsing of doc content |

### **ddlite_matcher.py**

Update coming...

### **ddlite.py**

#### Class: Relation

|**Member**|**Notes**|
|:---------|:--------|
| All `Sentence` members ||
|`uid`| Unique candidate ID |
| `prob` |  |
| `all_idxs` | |
| `labels` ||
| `xt` | XMLTree |
| `root` | XMLTree root|
|`tagged_sent` | Sentence text with matched tokens replaced by labels |
|`e1_idxs`| Tokens matched by first matcher |
|`e2_idxs`| Tokens matched by second matcher |
|`e1_label`| First matcher label |
|`e2_label`| Second matcher label |

|**Method**|**Notes**|
|:---------|:--------|
|`render` | Generates sentence dependency tree figure with matched tokens highlighted|
|`mention1(attribute='words')` | Return list of `attribute` tokens in first mention |
|`mention2(attribute='words')` | Return list of `attribute` tokens in second mention |
|`pre_window1(attribute='words', n=3)| Return list of `n` `attribute` tokens before first mention |
|`pre_window2(attribute='words', n=3)| Return list of `n` `attribute` tokens before second mention |
|`post_window1(attribute='words', n=3)| Return list of `n` `attribute` tokens after first mention |
|`post_window2(attribute='words', n=3)| Return list of `n` `attribute` tokens after second mention |

#### Class: Relations

|**Member**|**Notes**|
|:---------|:--------|
|`feats` | Feature matrix |
|`feat_index` | Feature names |

|**Method**|**Notes**|
|:---------|:--------|
|`__init__(content, matcher1=None, matcher2=None)`| `content` is a list of `Sentence` objects, or a path to Pickled `Relations` object |
|`[i]` | Access `i`th `Relation`|
|`len()` | Number of relations |
| `num_candidates()` |  |
| `num_feats()` |  |
| `extract_features(*args)` | |
| `dump_candidates(f)` | Pickle object to file |

#### Class: Entity

|**Member**|**Notes**|
|:---------|:--------|
| All `Sentence` members ||
|`uid`| Unique candidate ID |
| `prob` |  |
| `all_idxs` | |
| `labels` ||
| `xt` | XMLTree |
| `root` | XMLTree root|
|`tagged_sent` | Sentence text with matched tokens replaced by label |
|`idxs`| Tokens matched by matcher |
|`label`| Matcher label |

|**Method**|**Notes**|
|:---------|:--------|
|`render()` | Generates sentence dependency tree figure with matched tokens highlighted |
|`mention(attribute='words')` | Return list of `attribute` tokens in mention |
|`pre_window(attribute='words', n=3)| Return list of `n` `attribute` tokens before mention |
|`post_window(attribute='words', n=3)| Return list of `n` `attribute` tokens after mention |

#### Class: Entities

|**Member**|**Notes**|
|:---------|:--------|
|`feats` | Feature matrix |
|`feat_index` | Feature names |

|**Method**|**Notes**|
|:---------|:--------|
|`__init__(content, matcher=None)`| `content` is a list of `Sentence` objects, or a path to Pickled `Entities` object |
|`[i]` | Access `i`th `Entity`|
|`len()` | Number of entities |
| `num_candidates()` | |
| `num_feats()` |  |
| `extract_features(*args)` | |
| `dump_candidates(f)` | Pickle object to file |

#### Class: DDLiteModel (CandidateModel)

|**Member**|**Notes**|
|:---------|:--------|
|`C`| `Candidates` object |
|`feats`| Feature matrix |
|`logger`| `ModelLogger` object |
|`lf_matrix`| Labeling function matrix |
|`lf_names`| Labeling functions names |
|`X`| Joint LF and feature matrix |
|`w`| Learned weights|
|`gt`| `CandidateGT` object|
|`mindtagger_instance`| `MindTaggerInstance` object |
|`use_lfs` | Use LF weights for prediction?|
|`model` | Last model type learned |

|**Method**|**Notes**|
|:---------|:--------|
|`num_candidates()` ||
|`num_feats()` ||
|`num_lfs()` ||
|`gt_dictionary()` | Dictionary of ground truth labels indexed by candidate uid|
|`holdout()` | Indices of holdout set (validation and test) |
|`validation()` | Indices of validation set |
|`test()` |Indices of test set |
|`dev()` |Indices of dev set (subset of training with ground truth) |
|`dev1()` |Indices of primary dev set (for accuracy scores) |
|`dev2()` | Indices of secondary dev set (for generalization scores) |
|`training()` | Indices of training set |
|`set_holdout(idxs=None, validation_frac=0.5)` | Set holdout set to indices `idxs` or all candidates with ground truth (`idxs=None`); randomly split validation and test at fraction `validation_frac` |
|`get_labeled_ground_truth(subset=None)` |  Get indices and labels of candidates in `subset` with ground truth; `subset` can be `'training'`, `'validation'`, `'test'`, `None` (all), or an array of indices |
| `update_gt(gt, idxs=None, uids=None)` | Update ground truth with labels in `gt` array, indexed by `idxs` or candidate `uids`|
|`get_gt_dict()` | Return the dictionary of ground truth labels keyed by candidate uid |
|`set_lf_matrix(lf_matrix, names, clear=False)` | Set a custom LF matrix with LF names `names` ; appends to or clears existing based on `clear`|
|`apply_lfs(lfs_f, clear=False)` | Apply LFs in list `lfs_f`; appends to or clears existing based on `clear`|
|`delete_lf(lf)` | Delete LF by index or name |
|`print_lf_stats(idxs=None) | Print coverage, overlap, and conflict on the training set (`idxs=None`) or on candidates `idxs` |
|`plot_lf_stats()` | Generates plots to examine coverage, overlap, and conflict |
|`top_conflict_lfs(n=10)` | Show the top `n` conflict LFs |
|`lowest_coverage_lfs(n=10)` | Show the `n` LFs with lowest coverage |
|`lowest_empirical_accuracy_lfs(n=10)` | Show the `n` LFs with the lowest empirical accuracy against ground truth for candidates in the devset |
|`lf_summary_table()` | Print a table with summary statistics for each LF |
|`learn_weights(n_iter=1000, tol=1e-6, sample=False, n_samples=100, mu=None, n_mu=20, mu_min_ratio=1e-6, alpha=0, rate=0.01, decay=0.99, bias=False, warm_starts=False, use_sparse=True, verbose=False, log=True, plot=True)` | Use data programming to learn an elastic net logistic regression on candidates not in training set <ul><li>`n_iter`: maximum number of SGD iterations</li><li>`tol`: tolerance for relative gradient magnitude to weights for convergence</li><li>`sample`: Use batch SGD</li> <li>`n_samples`: Batch size for SGD</li><li>`mu`: sequence of regularization parameters to fit; if `None` and validation set exists, automatic sequence is chosen s.t. largest value is close to minimum value at which all weights are zero; if `None` and no validation set exists, uses a default parameter</li><li>`n_mu`: number of regularization parameters for automatic sequence</li><li>`mu_min_ratio`: lower bound for regularization parameter automatic sequence as ratio to largest value</li><li>`alpha`: elastic net mixing parameter (0 for l<sub>2</sub>, 1 for l<sub>1</sub>)</li><li>`rate`: SGD learning rate</li><li>`decay`: SGD learning rate decay</li><li>`bias`: use bias term</li><li>`warm_starts`: use warm starts when fitting a sequence of mu values</li><li>`use_sparse`: use sparse operations</li><li>`verbose`: be verbose</li><li>`log`: log result in `ModelLogger`</li><li>`plot`: show diagnostic plot for validation set performance</li></ul> |
|`learn_weights_validated(**kwargs)` | Wrapper for `learn_weights()`|
|`set_use_lfs(use=True)` | Setter for using LF weights in prediction |
|`get_log_odds(subset=None)` | Get log odds values for all candidates (`subset=None`), an index subset (`subset`), test set (`subset='test'`), or validation set (`subset='validation'`) |
|`get_predicted_probability(subset=None)` | Get predicted probabilities for candidates |
|`get_predicted(subset=None)` | Get predicted responses for candidates |
|`get_classification_accuracy(subset=None)` | Get classification accuracy over `subset` against ground truth |
|`plot_calibration()` | Show DeepDive calibration plots |
|`open_mindtagger(num_sample=None, abstain=False, **kwargs)` | Open MindTagger portal for labeling; sample is either the last sample (`num_sample=None`) or a random sample of size `num_sample`; if `abstain=True` show only canidates which haven't been labeled by any LF |
|`add_mindtagger_tags()` | Import tags from current MindTagger session |
|`add_to_log(log_id=None, subset='test', show=True)` | Add current learning result to log; precision and recall computed from ground truth in `subset` |
|`show_log(idx=None)` | Show all learning results (`idx=None`) or learning result `idx` | 

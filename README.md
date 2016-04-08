# DeepDive Lite

[![Build Status](https://travis-ci.org/HazyResearch/ddlite.svg?branch=master)](https://travis-ci.org/HazyResearch/ddlite)

## Motivation
DeepDive Lite provides a lighter-weight interface for creating a structured information extraction application in DeepDive. DeepDive Lite is built for rapid prototyping and development focused on **defining an input/output schema**, and **creating a set of _labeling functions_**. The goal is to be able to directly plug these objects into DeepDive proper to get a more scalable, performant, and customizable version of the application.

An immediate motivation is to provide a lighter-weight entry point to the DeepDive application development cycle for non-expert users. DeepDive Lite may also be useful for expert DeepDive users as a toolset for development and prototyping tasks.

DeepDive Lite is also part of a broader effort to answer the following research questions: 
* How much progress can be made with the _schema_ and _labeling functions_ being the only user entry points to the application development process?
* To what degree can DeepDive be seen/used as an _iterative compiler_, which takes in a rule-based program, and transforms it to a statistical learning and inference-based one?

## Installation / dependencies

<!-- TODO these manual instruction could be abstracted away with a simple launcher script, that takes as input a ipynb and simply opens it after any necessary setup.. -->

First of all, make sure all git submodules have been downloaded.

```bash
git submodule update --init
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

#### Class: SentenceParser

|**Method**|**Notes**|
|:---------|:--------|
|`__init()__`| Starts CoreNLPServer|
|`parse(doc, doc_id=None)` | Parse document into `Sentence`s|

#### Class: HTMLParser

|**Method**|**Notes**|
|:---------|:--------|
|`can_parse(f)`||
|`parse(f)`| Returns visible text in HTML file|

#### Class: TextParser

|**Method**|**Notes**|
|:---------|:--------|
|`can_parse(f)`||
|`parse(f)`| Returns all text in file|

#### Class: DocParser

|**Method**|**Notes**|
|:---------|:--------|
|`__init__(path, ftparser = TextParser())` | `path` can be single file, a directory, or a glob expression |
|`parseDocs()` | Returns docs as parsed by `ftparser` |
|`parseDocSentences()` | Returns `Sentence`s from `SentenceParser` parsing of doc content |

### **ddlite_matcher.py**

#### Class: DictionaryMatcher

|**Method**|**Notes**|
|:---------|:--------|
|`__init__(label, dictionary, match_attrib='words', ignore_case=True)`| |
|`apply(sentence)`| Tokens joined with spaces |

#### Class: RegexMatcher

|**Method**|**Notes**|
|:---------|:--------|
|`__init__(label, regex_pattern, match_attrib='words', ignore_case=True)`| Entire sentence text can be searched using `match_attrib='text'`|
|`apply(sentence)`| Tokens joined with spaces |

#### Class: MultiMatcher

|**Method**|**Notes**|
|:---------|:--------|
|`__init__(matcher1, matcher2,...)`| |
|`apply(sentence)`| Yields individual matcher label if `label` argument not in initialization |

### **ddlite.py**

#### Class: Relation

|**Member**|**Notes**|
|:---------|:--------|
| All `Sentence` members ||
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

#### Class: Relations

|**Member**|**Notes**|
|:---------|:--------|
|`feats` | Feature matrix |

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
| `prob` |  |
| `all_idxs` | |
| `labels` ||
| `xt` | XMLTree |
| `root` | XMLTree root|
|`tagged_sent` | Sentence text with matched tokens replaced by label |
|`e_idxs`| Tokens matched by matcher |
|`e_label`| Matcher label |

|**Method**|**Notes**|
|:---------|:--------|
|`render` | Generates sentence dependency tree figure with matched tokens highlighted|

#### Class: Entities

|**Member**|**Notes**|
|:---------|:--------|
|`feats` | Feature matrix |

|**Method**|**Notes**|
|:---------|:--------|
|`__init__(content, matcher=None)`| `content` is a list of `Sentence` objects, or a path to Pickled `Entities` object |
|`[i]` | Access `i`th `Entity`|
|`len()` | Number of entities |
| `num_candidates()` | |
| `num_feats()` |  |
| `extract_features(*args)` | |
| `dump_candidates(f)` | Pickle object to file |

#### Class: CandidateModel

|**Member**|**Notes**|
|:---------|:--------|
|`C`| `Candidates` object |
|`feats`| Feature matrix |
|`logger`| `ModelLogger` object |
|`LF`| Labeling function matrix |
|`LF_names`| Labeling functions names |
|`X`| Joint LF and feature matrix |
|`w`| Learned weights|
|`holdout`| Indices of holdout set |
|`mindtagger_instance`| `MindTaggerInstance` object |

|**Method**|**Notes**|
|:---------|:--------|
|`num_candidates()` ||
|`num_feats()` ||
|`num_LFs()` ||
|`set_gold_labels(gold)`| Set gold standard labels |
|`get_ground_truth(gt='resolve')` | Get ground truth from just MindTagger (`gt='mindtagger'`), just gold standard labels (`gt='gold'`), or resolve with conflict priority to gold standard (`gt='resolve'`) |
|`has_ground_truth()` | Get boolean array of candidates with some ground truth |
|`get_labeled_ground_truth(gt='resolve', subset=None)` |  Get indices and labels of candidates in `subset` labeled by method `gt`; `subset` can be `'holdout'` or an array of indices |
|`set_lf_matrix(lf_matrix, names, clear=False)` | Set a custom LF matrix with LF names `names` ; appends to or clears existing based on `clear`|
|`apply_lfs(lfs_f, clear=False)` | Apply LFs in list `lfs_f`; appends to or clears existing based on `clear`|
|`delete_lf(lf)` | Delete LF by index or name |
|`plot_lf_stats()` | Generates plots to examine coverage, overlap, and conflict |
|`top_conflict_lfs(n=10)` | Show the top `n` conflict LFs |
|`lowest_coverage_lfs(n=10)` | Show the `n` LFs with lowest coverage |
|`lowest_empirical_accuracy_lfs(n=10, subset=None)` | Show the `n` LFs with the lowest empirical accuracy against ground truth for candidates in `subset` |
|`set_holdout(idxs=None)` | Set holdout set to all candidates with ground truth (`idxs=None`) or subset `idxs`|
|`learn_weights(nfolds=3, maxIter=1000, tol=1e-6, sample=False, n_samples=100, mu=None, n_mu=20, mu_min_ratio=1e-6, alpha=0, opt_1se=True, use_sparse=True, plot=False, verbose=False, log=True)` | Learn an elastic net logistic regression on candidates not in holdout set <ul><li>`nfolds`: number of folds for cross-validation</li><li>`maxIter`: maximum number of SGD iterations</li><li>`tol`: tolerance for relative gradient magnitude to weights for convergence</li><li>`sample`: Use batch SGD</li> <li>`n_samples`: Batch size for SGD</li><li>`mu`: sequence of regularization parameters to fit; if `None`, automatic sequence is chosen s.t. largest value is close to minimum value at which all weights are zero</li><li>`n_mu`: number of regularization parameters for automatic sequence</li><li>`mu_min_ratio`: lower bound for regularization parameter automatic sequence as ratio to largest value</li><li>`alpha`: elastic net mixing parameter (0 for l<sub>2</sub>, 1 for l<sub>1</sub>)</li><li>`opt_1se`: use one-standard-error-rule for choosing regularization parameter</li><li>`use_sparse`: use sparse operations</li><li>`plot`: show diagnostic plot for cross-validation</li><li>`verbose`: be verbose</li><li>`log`: log result in `ModelLogger`</li></ul> |
|`get_link(subset=None)` | Get link function values for all candidates (`subset=None`), an index subset (`subset`), or holdout set (`subset='holdout'`) |
|`get_predicted_probability(subset=None)` | Get predicted probabilities for candidates |
|`get_predicted(subset=None)` | Get predicted responses for candidates |
|`get_classification_accuracy(gt='resolve', subset=None)` | Get classification accuracy over `subset` against ground truth determined by `gt` |
|`get_LF_priority_vote_accuracy(gt='resolve', subset=None)` | Get classification accuracy over `subset` against ground truth determined by `gt` from LFs alone; assignment prioritized by LF order |
|`plot_calibration()` | Show DeepDive calibration plots |
|`open_mindtagger(num_sample=None, ...)` | Open MindTagger portal for labeling; sample is either the last sample (`num_sample=None`) or a random sample of size `num_sample` |
|`add_mindtagger_tags()` | Import tags from current MindTagger session |
|`add_to_log(log_id=None, gt='resolve', subset='holdout', verb=True)` | Add current learning result to log; precision and recall computed from ground truth in `subset` determined by `gt` |
|`show_log(idx=None)` | Show all learning results (`idx=None`) or learning result `idx` | 

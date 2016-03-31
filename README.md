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

First of all, make sure all git submodule has been downloaded.

```bash
git submodule update --init
```

DeepDive Lite requires [a few python packages](python-package-requirement.txt) including:

* [nltk](http://www.nltk.org/install.html)
* [lxml](http://lxml.de/installation.html)
* [requests](http://docs.python-requests.org/en/master/user/install/#install)
* [numpy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
* [scipy](http://www.scipy.org/install.html)

We provide a simple way to install everything using `virtualenv`:

```bash
# set up a Python virtualenv
virtualenv .virtualenv
source .virtualenv/bin/activate

pip install --requirement python-package-requirement.txt
```

Alternatively, they could be installed system-wide if `sudo pip` is used instead of `pip` in the last command without the virtualenv setup and activation.

In addition the Stanford CoreNLP parser jars need to be downloaded; this can be done using:
```bash
./install-parser.sh
```

Finally, DeepDive Lite is built specifically with usage in **Jupyter/IPython notebooks** in mind.
The `jupyter` command is installed as part of the above installation steps, so the following command within the virtualenv opens our demo notebook.

```bash
jupyter notebook GeneTaggerExample_Extraction.ipynb
```

## Learning how to use DeepDive Lite
The best way to learn how to use is to open up the demo notebooks. **GeneTaggerExample_Extraction.ipynb** walks through the candidate extraction workflow for an entity tagging task. **GeneTaggerExample_Learning.ipynb** picks up where the extraction notebook left off. The learning notebook demonstrates the labeling function iteration workflow and learning methods.

## Best practices for labeling function iteration
The following flowchart illustrates the labeling function iteration workflow.

## Best practices for using DeepDive Lite notebooks
Here are a few practical tips for working with DeepDive Lite:
* Use `autoreload`
* Keep working source code in another file 
* Pickle extractions often and with unique names
* Document past labeling functions either remotely or with the `CandidateModel` log

## Documentation

**Class: Sentence**
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

**Class: SentenceParser**
|**Method**|**Notes**|
|:---------|:--------|
|`__init()__`| Starts CoreNLPServer|
|`parse(doc, doc_id=None)| Parse document into `Sentence`s|

**Class: DictionaryMatcher**
|**Method**|**Notes**|
|:---------|:--------|
|`__init__(label, dictionary, match_attrib='words', ignore_case=True)`| |
|`apply(sentence)`| Tokens joined with spaces |

**Class: RegexMatcher**
|**Method**|**Notes**|
|:---------|:--------|
|`__init__(label, regex_pattern, match_attrib='words', ignore_case=True)`| Entire sentence text can be searched using `match_attrib='text'`|
|`apply(sentence)`| Tokens joined with spaces |

**Class: MultiMatcher**
|**Method**|**Notes**|
|:---------|:--------|
|`__init__(matcher1, matcher2,...)`| |
|`apply(sentence)`| Yields individual matcher label if `label` argument not in initialization |



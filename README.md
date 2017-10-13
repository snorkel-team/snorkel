<img src="figs/logo_01.png" width="150"/>


**_v0.6.2_**


[![Documentation](https://readthedocs.org/projects/snorkel/badge/?version=master)](http://snorkel.readthedocs.io/en/master/)
[![Build Status](https://travis-ci.org/HazyResearch/snorkel.svg?branch=master)](https://travis-ci.org/HazyResearch/snorkel)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Acknowledgements
<img src="figs/darpa.JPG" width="80" height="80" align="middle" /> <img src="figs/ONR.jpg" width="100" height="80" align="middle" /> <img src="figs/moore_logo.png" width="100" height="60" align="middle" /> <img src="figs/nih_logo.png" width="80" height="60" align="middle" /> <img src="figs/mobilize_logo.png" width="100" height="60" align="middle" />

*Sponsored in part by DARPA as part of the [SIMPLEX](http://www.darpa.mil/program/simplifying-complexity-in-scientific-discovery) program under contract number N66001-15-C-4043 and also by the NIH through the [Mobilize Center](http://mobilize.stanford.edu/) under grant number U54EB020405.*

## Getting Started

* Installation instructions [below](#installation--dependencies)
* Get started with the tutorials [below](#learning-how-to-use-snorkel)
* Documentation [here](http://snorkel.readthedocs.io/en/master/)

## Motivation
Snorkel is a system for rapidly **creating, modeling, and managing training data**, currently focused on accelerating the development of _structured or "dark" data extraction applications_ for domains in which large labeled training sets are not available or easy to obtain.

Today's state-of-the-art machine learning models require massive labeled training sets--which usually do not exist for real-world applications. Instead, Snorkel is based around the new [data programming](https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly) paradigm, in which the developer focuses on writing a set of labeling functions, which are just scripts that programmatically label data. The resulting labels are noisy, but Snorkel automatically models this process—learning, essentially, which labeling functions are more accurate than others—and then uses this to train an end model (for example, a deep neural network in TensorFlow).

_Surprisingly_, by modeling a noisy training set creation process in this way, we can take potentially low-quality labeling functions from the user, and use these to train high-quality end models. We see Snorkel as providing a general framework for many [_weak supervision_](http://hazyresearch.github.io/snorkel/blog/weak_supervision.html) techniques, and as defining a new programming model for weakly-supervised machine learning systems.

## Users
We're lucky to have some amazing collaborators who are currently using Snorkel!

<img src="figs/user_logos.png" width="500" height="200" align="middle" />

However, **_Snorkel is very much a work in progress_**, so we're eager for any and all feedback... let us know what you think and how we can improve Snorkel in the [Issues](https://github.com/HazyResearch/snorkel/issues) section!

## References

### Best References:
* **_[Data Programming: Creating Large Training Sets, Quickly](https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly)_ (NIPS 2016)**
* **_[Learning the Structure of Generative Models without Labeled Data](https://arxiv.org/abs/1703.00854)_ (ICML 2017)**
* **_[Snorkel: Fast Training Set Generation for Information Extraction](http://hazyresearch.github.io/snorkel/pdfs/snorkel_demo.pdf)_ (SIGMOD DEMO 2017)**
* _[Inferring Generative Model Structure with Static Analysis](https://arxiv.org/abs/1709.02477)_ (NIPS 2017)
* _[Data Programming with DDLite: Putting Humans in a Different Part of the Loop](http://cs.stanford.edu/people/chrismre/papers/DDL_HILDA_2016.pdf)_ (HILDA @ SIGMOD 2016; note Snorkel was previously <em>DDLite</em>)
* _[Socratic Learning: Correcting Misspecified Generative Models using Discriminative Models](https://arxiv.org/abs/1610.08123)_
* _[Fonduer: Knowledge Base Construction from Richly Formatted Data](https://arxiv.org/abs/1703.05028)_

### Further Reading:
* _[Learning to Compose Domain-Specific Transformations for Data Augmentation](https://arxiv.org/abs/1709.01643)_ (NIPS 2017)
* _[Gaussian Quadrature for Kernel Features](https://arxiv.org/abs/1709.02605)_ (NIPS 2017)

## Learning how to use Snorkel
The [introductory tutorial](https://github.com/HazyResearch/snorkel/tree/master/tutorials/intro) covers the entire Snorkel workflow, showing how to extract spouse relations from news articles.
The tutorial is available in the following directory:
```
tutorials/intro
```

## Release Notes
### Major changes in v0.6:
* Support for categorical classification, including "dynamically-scoped" or _blocked_ categoricals (see [tutorial](tutorials/advanced/Categorical_Classes.ipynb))
* Support for structure learning (see [tutorial](tutorials/advanced/Structure_Learning.ipynb), ICML 2017 paper)
* Support for labeled data in generative model
* Refactor of TensorFlow bindings; fixes grid search and model saving / reloading issues (see `snorkel/learning`)
* New, simplified Intro tutorial ([here](tutorials/intro))
* Refactored parser class and support for [spaCy](https://spacy.io/) as new default parser
* Support for easy use of the [BRAT annotation tool](http://brat.nlplab.org/) (see [tutorial](tutorials/advanced/BRAT_Annotations.ipynb))
* Initial Spark integration, for scale out of LF application (see [tutorial](tutorials/snark/Snark%20Tutorial.ipynb))
* Tutorial on using crowdsourced data [here](tutorials/crowdsourcing/Crowdsourced_Sentiment_Analysis.ipynb)
* Integration with [Apache Tika](http://tika.apache.org/) via the [Tika Python](http://github.com/chrismattmann/tika-python.git) binding.
* And many more fixes, additions, and new material!

## Installation
Snorkel uses Python 2.7 and requires [a few python packages](python-package-requirement.txt) which can be installed using [`conda`](https://www.continuum.io/downloads) and `pip`.

### Setting Up Conda
Installation is easiest if you download and install [`conda`](https://www.continuum.io/downloads).
If you are running multiple version of Python, you might need to run:
```
conda create -n py2Env python=2.7 anaconda
```
And then run the correct environment:
```
source activate py2Env
```

### Installing dependencies
First install [NUMBA](https://numba.pydata.org/), a package for high-performance numeric computing in Python via Conda:
```bash
conda install numba
```

Then install the remaining package requirements:
```bash
pip install --requirement python-package-requirement.txt
```

Finally, enable `ipywidgets`:
```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

_Note: If you are using conda and experience issues with `lxml`, try running `conda install libxml2`._

_Note: Currently the `Viewer` is supported on the following versions:_
* `jupyter`: 4.1
* `jupyter notebook`: 4.2

In some tutorials, etc. we also use [Stanford CoreNLP](http://stanfordnlp.github.io/CoreNLP/) for pre-processing text; you will be prompted to install this when you run `run.sh`.

### Frequently Asked Questions
See [this FAQ](https://hazyresearch.github.io/snorkel/install_faq) for help with common questions that arise.

## Running
After installing, just run:
```
./run.sh
```

## Issues
We like [issues](https://github.com/HazyResearch/snorkel/issues) as a place to put bugs, questions, feature requests, etc- don't be shy!
If submitting an issue about a bug, however, **please provide a pointer to a notebook (and relevant data) to reproduce it.**

*Note: if you have an issue with the matplotlib install related to the module `freetype`, see [this post](http://stackoverflow.com/questions/20533426/ubuntu-running-pip-install-gives-error-the-following-required-packages-can-no); if you have an issue installing ipython, try [upgrading setuptools](http://stackoverflow.com/questions/35943606/error-on-installing-ipython-for-python-3-sys-platform-darwin-and-platform)*

## Jupyter Notebook Best Practices

Snorkel is built specifically with usage in **Jupyter/IPython notebooks** in mind; an incomplete set of best practices for the notebooks:

It's usually most convenient to write most code in an external `.py` file, and load as a module that's automatically reloaded; use:
```python
%load_ext autoreload
%autoreload 2
```
A more convenient option is to add these lines to your IPython config file, in `~/.ipython/profile_default/ipython_config.py`:
```
c.InteractiveShellApp.extensions = ['autoreload']     
c.InteractiveShellApp.exec_lines = ['%autoreload 2']
```

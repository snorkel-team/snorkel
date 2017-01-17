<img src="figs/logo_01.png" width="150"/>
**_v0.4.0_**

[![Documentation](https://readthedocs.org/projects/snorkel/badge/?version=master)](http://snorkel.readthedocs.io/en/latest/)
[![Build Status](https://travis-ci.org/HazyResearch/snorkel.svg?branch=master)](https://travis-ci.org/HazyResearch/snorkel)

## Acknowledgements
<img src="figs/darpa.JPG" width="80" height="80" />
<img src="figs/ONR.jpg" width="100" height="80" />

*Sponsored in part by DARPA as part of the [SIMPLEX](http://www.darpa.mil/program/simplifying-complexity-in-scientific-discovery) program under contract number N66001-15-C-4043.*

## Getting Started

* [Data Programming: ML with Weak Supervision](http://hazyresearch.github.io/snorkel/blog/weak_supervision.html)
* Installation instructions [below](#installation--dependencies)
* Get started with the tutorials [below](#learning-how-to-use-snorkel)
* Documentation [here](http://snorkel.readthedocs.io/en/latest/)

## Motivation
Snorkel is intended to be a lightweight but powerful framework for developing **structured information extraction applications** for domains in which large labeled training sets are not available or easy to obtain, using the _data programming_ paradigm.

In the data programming approach to developing a machine learning system, the developer focuses on writing a set of _labeling functions_, which create a large but noisy training set. Snorkel then learns a generative model of this noise&mdash;learning, essentially, which labeling functions are more accurate than others&mdash;and uses this to train a discriminative classifier.

At a high level, the idea is that developers can focus on writing labeling functions&mdash;which are just (Python) functions that provide a label for some subset of data points&mdash;and not think about algorithms _or_ features!

**_Snorkel is very much a work in progress_**, but some people have already begun developing applications with it, and initial feedback has been positive... let us know what you think, and how we can improve it, in the [Issues](https://github.com/HazyResearch/snorkel/issues) section!

### References
* Data Programming, to appear at NIPS 2016: [https://arxiv.org/abs/1605.07723](https://arxiv.org/abs/1605.07723)
* Workshop paper from HILDA 2016 (note Snorkel was previously _DDLite_): [here](http://cs.stanford.edu/people/chrismre/papers/DDL_HILDA_2016.pdf)

## Installation / dependencies

Snorkel uses Python 2.7 and requires [a few python packages](python-package-requirement.txt) which can be installed using `pip`:
```bash
pip install --requirement python-package-requirement.txt
```
Note that `sudo` can be prepended to install dependencies system wide if this is an option and the above does not work.

Snorkel currently relies on `numba`, which occasionally requires a bit more work to install! One option is to use [`conda`](https://www.continuum.io/downloads). If installing manually, you may just need to make sure the right version of `llvmlite` and LLVM is installed and used; for example on Ubuntu, run:
```bash
apt-get install llvm-3.8
LLVM_CONFIG=/usr/bin/llvm-config-3.8 pip install llvmlite
LLVM_CONFIG=/usr/bin/llvm-config-3.8 pip install numba
```

Finally, enable `ipywidgets`:
```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

_Note: Currently the `Viewer` is supported on the following versions:_
* `jupyter`: 4.1
* `jupyter notebook`: 4.2

By default (e.g. in the tutorials, etc.) we also use [Stanford CoreNLP](http://stanfordnlp.github.io/CoreNLP/) for pre-processing text; you will be prompted to install this when you run `run.sh`.

Alternatively, `virtualenv` can be used by starting with:
```bash
virtualenv -p python2.7 .virtualenv
source .virtualenv/bin/activate
```

## Running
After installing (see below), just run:
```
./run.sh
```

## Learning how to use Snorkel
The [introductory tutorial](https://github.com/HazyResearch/snorkel/tree/master/tutorials/intro) covers the entire Snorkel workflow, showing how to extract spouse relations from news articles.
The tutorial is available in the following directory:
```
tutorials/intro
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

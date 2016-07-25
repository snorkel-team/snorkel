# Snorkel

[![Build Status](https://travis-ci.org/HazyResearch/snorkel.svg?branch=master)](https://travis-ci.org/HazyResearch/snorkel)

## Acknowledgements
<img src="figs/darpa.JPG" width="80" height="80" />
Sponsored in part by DARPA as part of the [SIMPLEX](http://www.darpa.mil/program/simplifying-complexity-in-scientific-discovery) program under contract number N66001-15-C-4043.

## Versions
* `master`: Current stable development version (v0.3)
* `v0.2-stable`: Last full version (v0.2)

## Running
After installing (see below), just run:
```
./run.sh
```

## Motivation
Snorkel is intended to be a lightweight but powerful framework for developing **structured information extraction applications** for domains in which large labeled training sets are not available or easy to obtain, using the _data programming_ paradigm.

In the data programming approach to developing a machine learning system, the developer focuses on writing a set of _labeling functions_, which create a large but noisy training set. Snorkel then learns a generative model of this noise&mdash;learning, essentially, which labeling functions are more accurate than others&mdash;and uses this to train a discriminative classifier.

At a high level, the idea is that developers can focus on writing labeling functions&mdash;which are just (Python) functions that provide a label for some subset of data points&mdash;and not think about algorithms _or_ features!

**_Snorkel is very much a work in progress_**, but some people have already begun developing applications with it, and initial feedback has been positive... let us know what you think, and how we can improve it, in the [Issues](https://github.com/HazyResearch/ddlite/issues) section!

### References
* Technical report on Data Programming: [https://arxiv.org/abs/1605.07723](https://arxiv.org/abs/1605.07723)
* Workshop paper from HILDA 2016 (note Snorkel was previously _DDLite_): [here](http://cs.stanford.edu/people/chrismre/papers/DDL_HILDA_2016.pdf)

## Installation / dependencies

Snorkel requires [a few python packages](python-package-requirement.txt) including:

* [nltk](http://www.nltk.org/install.html)
* [lxml](http://lxml.de/installation.html)
* [requests](http://docs.python-requests.org/en/master/user/install/#install)
* [numpy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html)
* [scipy](http://www.scipy.org/install.html)
* [matplotlib](http://matplotlib.org/users/installing.html)
* [theano](http://deeplearning.net/software/theano/install.html)

We provide a simple way to install everything using `virtualenv`:

```bash
# set up a Python virtualenv
virtualenv .virtualenv
source .virtualenv/bin/activate

pip install --requirement python-package-requirement.txt
```

Finally, enable `ipywidgets`:
```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

*Note: if you have an issue with the matplotlib install related to the module `freetype`, see [this post](http://stackoverflow.com/questions/20533426/ubuntu-running-pip-install-gives-error-the-following-required-packages-can-no); if you have an issue installing ipython, try [upgrading setuptools](http://stackoverflow.com/questions/35943606/error-on-installing-ipython-for-python-3-sys-platform-darwin-and-platform)*

Alternatively, they could be installed system-wide if `sudo pip` is used instead of `pip` in the last command without the virtualenv setup and activation.

## Learning how to use Snorkel
New tutorial (in progress; covers through candidate extraction for entities):
```
tutorial/CDR_tutorial.ipynb
```

Supported legacy tutorial (covers full pipeline):
 * **[GeneTaggerExample_Extraction.ipynb](https://github.com/HazyResearch/ddlite/blob/master/examples/GeneTaggerExample_Extraction.ipynb)** walks through the candidate extraction workflow for an entity tagging task. * **[GeneTaggerExample_Learning.ipynb](https://github.com/HazyResearch/ddlite/blob/master/examples/GeneTaggerExample_Learning.ipynb)** picks up where the extraction notebook left off. The learning notebook demonstrates the labeling function iteration workflow and learning methods.

## Documentation
To generate documentation (built using [pdoc](https://github.com/BurntSushi/pdoc)), run `./generate_docs.sh`.

## Issues
We like [issues](https://github.com/HazyResearch/snorkel/issues) as a place to put bugs, questions, feature requests, etc- don't be shy!
If submitting an issue about a bug, however, **please provide a pointer to a notebook (and relevant data) to reproduce it.**

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

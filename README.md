<img src="figs/logo_01.png" width="150"/>


**_v0.5.0_**


[![Documentation](https://readthedocs.org/projects/snorkel/badge/?version=master)](http://snorkel.readthedocs.io/en/latest/)
[![Build Status](https://travis-ci.org/HazyResearch/snorkel.svg?branch=master)](https://travis-ci.org/HazyResearch/snorkel)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Acknowledgements
<img src="figs/darpa.JPG" width="80" height="80" align="middle" />
<img src="figs/ONR.jpg" width="100" height="80" align="middle" />
<img src="figs/moore_logo.png" width="100" height="60" align="middle" />
<img src="figs/nih_logo.png" width="80" height="60" align="middle" />
<img src="figs/mobilize_logo.png" width="100" height="60" align="middle" />

*Sponsored in part by DARPA as part of the [SIMPLEX](http://www.darpa.mil/program/simplifying-complexity-in-scientific-discovery) program under contract number N66001-15-C-4043 and also by the NIH through the [Mobilize Center](http://mobilize.stanford.edu/) under grant number U54EB020405.*

## Getting Started

* Installation instructions [below](#installation--dependencies)
* Get started with the tutorials [below](#learning-how-to-use-snorkel)
* Documentation [here](http://snorkel.readthedocs.io/en/latest/)

## Motivation
Snorkel is a system for rapidly **creating, modeling, and managing training data**, currently focused on accelerating the development of _structured or "dark" data extraction applications_ for domains in which large labeled training sets are not available or easy to obtain.

Today's state-of-the-art machine learning models require massive labeled training sets--which usually do not exist for real-world applications. Instead, Snorkel is based around the new [data programming](https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly) paradigm, in which the developer focuses on writing a set of labeling functions, which are just scripts that programmatically label data. The resulting labels are noisy, but Snorkel automatically models this process—learning, essentially, which labeling functions are more accurate than others—and then uses this to train an end model (for example, a deep neural network in TensorFlow).

_Surprisingly_, by modeling a noisy training set creation process in this way, we can take potentially low-quality labeling functions from the user, and use these to train high-quality end models. We see Snorkel as providing a general framework for many [_weak supervision_](http://hazyresearch.github.io/snorkel/blog/weak_supervision.html) techniques, and as defining a new programming model for weakly-supervised machine learning systems.

### Users
We're lucky to have some amazing collaborators who are currently using Snorkel!

<img src="figs/user_logos.png" width="500" height="200" align="middle" />

However, **_Snorkel is very much a work in progress_**, so we're eager for any and all feedback... let us know what you think and how we can improve Snorkel in the [Issues](https://github.com/HazyResearch/snorkel/issues) section!

### References
* _Data Programming: Creating Large Training Sets, Quickly_, ([NIPS 2016](https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly))
* _Data Programming with DDLite: Putting Humans in a Different Part of the Loop_, ([HILDA @ SIGMOD 2016](http://cs.stanford.edu/people/chrismre/papers/DDL_HILDA_2016.pdf))
* _Snorkel: A System for Lightweight Extraction_, ([CIDR 2017](http://cidrdb.org/cidr2017/gongshow/abstracts/cidr2017_73.pdf))
* Data Programming: ML with Weak Supervision ([blog](http://hazyresearch.github.io/snorkel/blog/weak_supervision.html))
* _Learning the Structure of Generative Models without Labeled Data_, ([preprint](https://arxiv.org/abs/1703.00854))
* _Fonduer: Knowledge Base Construction from Richly Formatted Data_, ([preprint](https://arxiv.org/abs/1703.05028), [blog](https://hazyresearch.github.io/snorkel/blog/fonduer.html))

## Installation / dependencies

Snorkel uses Python 2.7 and requires [a few python packages](python-package-requirement.txt) which can be installed using `pip`:
```bash
pip install --requirement python-package-requirement.txt
```
If a package installation fails, then all of the packages below it in `python-package-requirement.txt` will fail to install as well. This can be avoided by running the following command instead of the above:
```bash
cat python-package-requirement.txt | xargs -n 1 pip install
```
Note that you may have to run `pip2` if you have Python3 installed on your system, and that `sudo` can be prepended to install dependencies system wide if this is an option and the above does not work.
For some pointers on difficulties in using `source` in shell, see [Issue 506](https://github.com/HazyResearch/snorkel/issues/506).

Finally, enable `ipywidgets`:
```bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
```

_Note: Currently the `Viewer` is supported on the following versions:_
* `jupyter`: 4.1
* `jupyter notebook`: 4.2

By default (e.g. in the tutorials, etc.) we also use [Stanford CoreNLP](http://stanfordnlp.github.io/CoreNLP/) for pre-processing text; you will be prompted to install this when you run `run.sh`.

### Working with Conda
One great option, which can make installation and use easier, is to use [`conda`](https://www.continuum.io/downloads).
If you are running multiple version of Python, you might need to run:
```
conda create -n py2Env python=2.7 anaconda
```
And then run the correct environment:
```
source activate py2Env
```

### Installing Numbskull + NUMBA
Snorkel currently relies on [`numbskull`](https://github.com/HazyResearch/numbskull) and `numba`, which occasionally requires a bit more work to install! One option is to use [`conda`](https://www.continuum.io/downloads) as above. If installing manually, you may just need to make sure the right version of `llvmlite` and LLVM is installed and used; for example on Ubuntu, run:
```bash
apt-get install llvm-3.8
LLVM_CONFIG=/usr/bin/llvm-config-3.8 pip install llvmlite
LLVM_CONFIG=/usr/bin/llvm-config-3.8 pip install numba
```
and on Mac OSX, one option is to use homebrew as follows:
```
brew install llvm38 --with-rtti
LLVM_CONFIG=/usr/local/Cellar/llvm\@3.8/3.8.1/bin/llvm-config-3.8 pip install llvmlite
LLVM_CONFIG=/usr/local/Cellar/llvm\@3.8/3.8.1/bin/llvm-config-3.8 pip install numba
```
Finally, once `numba` is installed, re-run the `numbskull` install from the `python-package-requirement.txt` script:
```
pip install git+https://github.com/HazyResearch/numbskull@dev
```
### Using virtualenv
Alternatively, `virtualenv` can be used by starting with:
```bash
virtualenv -p python2.7 .virtualenv
source .virtualenv/bin/activate
```
If you have issues using Jupyter notebooks with virualenv, see [this tutorial](http://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs)


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

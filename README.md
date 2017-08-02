## Containerized snorkel
This fork of the snorkel repo has been modified slightly to be build and deployed in a Docker container. No clone of this repository is necessary, that will be performed in a the Docker image build. Just download the app in the [*snocker* directory](https://github.com/DrPinkACN/snorkel/tree/master/snocker/app) or create a local dockerfile and copy the contents into that file.

Make sure Docker is installed on your run environment and that the Dockerfile is available on the run envinment in a directory named app. Then build the image.

```
$> docker build app -t snocker:0.6.2a
```

Now you can run the app. Link the home directory to somewhere easy to find.

```
$> docker run -it --name snocker -p 8887:8887 -v ~/some/local/dir/mapped/to/home:/home snocker:0.6.2a
## hit esc key sequence: ctrl+p+q
```

You can now execute a bash command in the running container, move the snorkel directory, and run snorkel.

```
$> docker exec -it snocker bash
#> mv /snorkel /home
#> cd /home/snorkel
#> ./run.sh
```

Feel free to install CoreNLP if you plan to use that parser instead of spaCy. After the install runs, Jupyter Notebook will start and you will be prompted with a dialog asking you to copy and paste a url... something like this:

```
    Copy/paste this URL into your browser when you connect for the first time,
    to login with a token:
        http://0.0.0.0:8887/?token=b40203286f7c49e021c0dc0767239129f6865ce83b93a559
```

**Copy just the token and direct your web browser to port 8887 on the server running Docker**. If running on your local machine, this will look something like this: `http://localhost:8887/`

You will be prompted to paste the token in as a password the first time you visit this Jupyter Notebook instance. After that the running container will remeber you.

That's pretty much it.

I find the [Docker cheatsheet](https://www.docker.com/sites/default/files/Docker_CheatSheet_08.09.2016_0.pdf) to be a pretty useful reference.

<img src="figs/logo_01.png" width="150"/>


**_v0.6.2_**


[![Documentation](https://readthedocs.org/projects/snorkel/badge/?version=master)](http://snorkel.readthedocs.io/en/master/)
[![Build Status](https://travis-ci.org/HazyResearch/snorkel.svg?branch=master)](https://travis-ci.org/HazyResearch/snorkel)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Acknowledgements
The Hazy Research group at Stanford University is the initial source of [this repository](https://github.com/HazyResearch/snorkel)

## Getting Started

* Get started with the tutorials [below](#learning-how-to-use-snorkel)
* Documentation [here](http://snorkel.readthedocs.io/en/master/)

## Motivation
Snorkel is a system for rapidly **creating, modeling, and managing training data**, currently focused on accelerating the development of _structured or "dark" data extraction applications_ for domains in which large labeled training sets are not available or easy to obtain.

Today's state-of-the-art machine learning models require massive labeled training sets--which usually do not exist for real-world applications. Instead, Snorkel is based around the new [data programming](https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly) paradigm, in which the developer focuses on writing a set of labeling functions, which are just scripts that programmatically label data. The resulting labels are noisy, but Snorkel automatically models this process—learning, essentially, which labeling functions are more accurate than others—and then uses this to train an end model (for example, a deep neural network in TensorFlow).

_Surprisingly_, by modeling a noisy training set creation process in this way, we can take potentially low-quality labeling functions from the user, and use these to train high-quality end models. We see Snorkel as providing a general framework for many [_weak supervision_](http://hazyresearch.github.io/snorkel/blog/weak_supervision.html) techniques, and as defining a new programming model for weakly-supervised machine learning systems.

## References
* _[Data Programming: Creating Large Training Sets, Quickly](https://papers.nips.cc/paper/6523-data-programming-creating-large-training-sets-quickly)_ (NIPS 2016)
* _[Snorkel: Fast Training Set Generation for Information Extraction](http://hazyresearch.github.io/snorkel/pdfs/snorkel_demo.pdf)_ (SIGMOD DEMO 2017)
* _[Learning the Structure of Generative Models without Labeled Data](https://arxiv.org/abs/1703.00854)_ (ICML 2017)
* _[Data Programming with DDLite: Putting Humans in a Different Part of the Loop](http://cs.stanford.edu/people/chrismre/papers/DDL_HILDA_2016.pdf)_ (HILDA @ SIGMOD 2016; note Snorkel was previously <em>DDLite</em>)
* _[Socratic Learning: Correcting Misspecified Generative Models using Discriminative Models](https://arxiv.org/abs/1610.08123)_
* _[Fonduer: Knowledge Base Construction from Richly Formatted Data](https://arxiv.org/abs/1703.05028)_

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
This is taken care of with the Dockerfile instructions above

_Note: Currently the `Viewer` is supported on the following versions:_ **Sooooo this might not work**
* `jupyter`: 4.1
* `jupyter notebook`: 4.2

If you want to view and annotate you can use Brat. In addition, I will be documenting some directions on how to do this with library I built on top of spaCy called [capsule](https://github.com/DrPinkACN/Capsule).

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

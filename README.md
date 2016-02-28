# DeepDive Lite

## Motivation
DeepDive Lite is an attempt to provide a lighter-weight interface to the process of creating a structured information extraction application in DeepDive.  DeepDive Lite is built for rapid prototyping and development solely focused around **defining an input/output schema**, and **creating a set of _distant supervision rules_**.  The goal is to then be able to directly plug these objects into DeepDive proper, and instantly get a more scalable, performant and customizable version of the application (which can then be iterated on within the DeepDive development framework).

One shorter-term motivation is also to provide a lighter-weight entry point to the DeepDive application development cycle for new non-expert users.  However DeepDive Lite may also be useful for "expert" DeepDive users as a simple toolset for certain development and prototyping tasks.

DeepDive Lite is also part of a broader attempt to answer the following research questions: how much progress can be made with the _schema_ and _distant supervision rules_ being the sole user entry point to the application development process?  To what degree can DeepDive be seen/used as an (iterative) _compiler_, which takes in a rule-based program, and transforms it to a statistical learning & inference-based one?

## Installation / dependencies
DeepDive Lite requires the following python modules; we provide example install commands using `pip`:
* [nltk](http://www.nltk.org/install.html): `sudo pip install -U nltk`
* [lxml](http://lxml.de/installation.html): `sudo pip install -U lxml`
* [numpy](http://docs.scipy.org/doc/numpy-1.10.1/user/install.html): `sudo pip install -U numpy`
* [scipy](http://www.scipy.org/install.html)

In addition the Stanford CoreNLP parser jars need to be downloaded; this can be done using:
```bash
./install-parser.sh
```

Finally, DeepDive Lite is built specifically with usage in **Jupyter/IPython notebooks** in mind; see their [installation instructions](http://jupyter.readthedocs.org/en/latest/install.html).

## Basics
Please see the Jupyter notebook demo in `DeepDiveLite.ipynb` for more detail!

### Preprocessing Input

### Candidate Extraction

### Feature Extraction

### Learning

# DeepDive Lite

## Motivation
DeepDive Lite is an attempt to provide a lighter-weight interface to the process of creating a structured information extraction application in DeepDive.  DeepDive Lite is built for rapid prototyping and development solely focused around **defining an input/output schema**, and **creating a set of _distant supervision rules_**.  The goal is to then be able to directly plug these objects into DeepDive proper, and instantly get a more scalable, performant and customizable version of the application (which can then be iterated on within the DeepDive development framework).

One shorter-term motivation is also to provide a lighter-weight entry point to the DeepDive application development cycle for new non-expert users.  However DeepDive Lite may also be useful for "expert" DeepDive users as a simple toolset for certain development and prototyping tasks.

DeepDive Lite is also part of a broader attempt to answer the following research questions: how much progress can be made with the _schema_ and _distant supervision rules_ being the sole user entry point to the application development process?  To what degree can DeepDive be seen/used as an (iterative) _compiler_, which takes in a rule-based program, and transforms it to a statistical learning & inference-based one?

## Installation / dependencies
DeepDive Lite requires the following python modules; we provide example install commands using `pip`:
* [nltk](http://www.nltk.org/install.html): `sudo pip install -U nltk`
* [lxml](http://lxml.de/installation.html): `sudo pip install -U lxml`
* [requests](http://lxml.de/installation.html): `sudo pip install -U requests`
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
The `SentenceParser` can be used to split a document (input as a string) into sentences, and to extract a range of basic linguistic features from these sentences, such as part-of-speech tags, a dependency tree parse, lemmatized words, etc:
```python
parser = SentenceParser()
for sent in parser.parse(doc_string):
  yield sent
```

The output is a generator of `Sentence` objects, which have various useful sentence attributes (as mentioned partially above).

**_Note:_** this is often the slowest part of the process, so for large document sets, pre-processing with high parallelism and/or external to DeepDive Lite is recommended.  Further improvements on speed to come as well [TODO].


### Candidate Extraction
DeepDive Lite is (currently) focused around extracting _relation mentions_ from text, involving either one or two entities.  In either case, we define a `Relations` object, which extracts a set of _candidate relation mentions_.  Our task is then to train the system to distinguish _true_ relation mentions from _false_ ones.

For the binary case, we define a relation based on two sets of _entity mentions_, described via declarative operators.  For example, we can define a relation as occuring between phrases that match a list of gene names, and phrases that match a list of phenotype names, and then extract them from a set of sentences:
```python
r = Relations(
  DictionaryMatch('G', genes, ignore_case=False),
  DictionaryMatch('P', phenos),
  sentences)
```
The `Relations` object both extracts the candidate relations, and then serves as the interface to and container of them.  To access them- as `Relation` objects- we use `r.relations`, and can render a visualization of one via e.g. `r.relations[0].render`:
![rendered-relation](rel_tree.png)

### Distant Supervision
The goal is now to create a set of _rules_ that specify which relations are true versus false, which we will use to _train_ the system to perform this inference correctly.*

In the context of DeepDive Lite, **a rule is simply a function which accepts a `Relation` object and returns a value in {-1,0,1}** (where 0 means 'abstain').  Once a list of rules is created, this list is applied to the `Relations` set via `r.apply_rules(rules)`.  This generates a matrix of rule labels `r.rules`, with rows corresponding to rules, and columns to relation candidates.

Note also that a natural question is: 'how well would my rules alone do on the classification task?'.  This provides a natural baseline for assessing  further performance downstream.  To answer this question, relative to a set of ground truth, we can use `r.get_rule_priority_vote_accuracy(idxs, ground_truth)`.

*_Note that if a set of labeled data is available, these labels could technically be used to create a trivial set of rules; however we assume we are operating in domains where a sufficiently large labeled training set is not available._

### Feature Extraction
Feature extraction is done _automatically_ via `r.extract_features()`.  The method of featurization can however be selected and customized [TODO].  After this has been performed, a (sparse) matrix of features `r.feats` is generated, with rows corresponding to features and columns to relation candidates.

### Learning
Learning of rule & feature weights can be done using logistic regression, via `r.learn_feats_and_weights()`.  This generates a learned parameter array `r.w`.  Predicted relation values (with -1 meaning false, and 1 meaning true) can then be generated via `r.get_predicted`, and accuracy relative to a set of ground truth labels via `r.get_classification_accuracy(idxs, ground_truth)`.

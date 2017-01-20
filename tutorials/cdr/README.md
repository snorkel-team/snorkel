# Extracting Chemical-Disease Relations from Academic Literature

In this advanced tutorial, we will build a Snorkel application to tackle the more
challenging task of identifying mentions of a chemical causing a disease from
academic literature. The task is inspired by the 
[BioCreative CDR task from 2015](http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/),
which is also where the data is derived from.
The tutorial is broken up into three notebooks, each covering a step in the pipeline:

1. **Preprocessing and Candidate Extraction [[CDR_Tutorial_1](CDR_Tutorial_1.ipynb)]:**
First we parse the raw input documents into _contexts_ (documents, sentences), and extract
consituent linguistic attributes. We also use _matcher_ operators to extract sets of 
_candidate_ chemical-disease relation mentions from the preprocessed input using entity tags
from an automated tool.

2. **Developing and Modeling Labeling Functions [[CDR_Tutorial_2](CDR_Tutorial_2.ipynb)]:**
We develop a more advanced set of labeling functions, based on text patterns and distant 
supervision. These labeling functions also have dependencies between them, and we model
these automatically. Finally, we train a generative model with the learned dependencies.

3. **Learning an Extraction Model[[CDR_Tutorial_3](CDR_Tutorial_3.ipynb)]:**
Here, we go through the process of writing _labeling functions_, learning a generative
model over them, using the generative model to train a _noise-aware_ discriminative
model to make predictions over the candidates, and evaluating the discriminative model
on the development candidate set.

## Example

For example, in the sentence
> Warfarin-induced artery calcification is accelerated by growth and vitamin D.

our goal is to extract the chemical-disease relation pair
("warfarin", "artery calcification").

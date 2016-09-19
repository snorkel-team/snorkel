# Chemical-Disease Relation (CDR) Tutorial

In this tutorial, we'll be writing an application to extract *mentions of*
**chemical-induced-disease relationships** from Pubmed abstracts, as per the 
[BioCreative CDR Challenge](http://www.biocreative.org/resources/corpora/biocreative-v-cdr-corpus/).
We will go through the following procedure, with corresponding notebooks:

1. **Preprocessing [[CDR_Tutorial_1](CDR_Tutorial_1.ipynb)]:** First we parse the raw input documents into
_contexts_ (documents, sentences), and extract consituent linguistic attributes.

2. **Candidate Extraction [[CDR_Tutorial_2](CDR_Tutorial_2.ipynb)]:** Next, we use simple _matcher_ operators
to extract a set of _candidate_ CDR relation mentions from the preprocessed input.

3. **Creating and/or Loading Test Labels [[CDR_Tutorial_3](CDR_Tutorial_3.ipynb)]:** Next, we use the `Viewer`
to label a test set to evaluate against, and/or use helpers to load external test labels.

4. **Learning [[CDR_Tutorial_4](CDR_Tutorial_4.ipynb)]:** Here, we go through the process of writing _labeling
functions_, learning a generative model over them, and then using this to train a _noise-aware_ discriminative
model to make predictions over the candidates.

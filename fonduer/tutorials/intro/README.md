# Extracting Spouse Relations from the News

In this tutorial, we will walk through the process of using Snorkel to identify mentions of
spouses in a corpus of news articles.
The tutorial is broken up into 5 notebooks, each covering a step in the pipeline:

1. **Preprocessing [[Intro_Tutorial_1](Intro_Tutorial_1.ipynb)]:**
First we parse the raw input documents into _contexts_ (documents, sentences), and extract
consituent linguistic attributes.

2. **Candidate Extraction [[Intro_Tutorial_2](Intro_Tutorial_2.ipynb)]:**
Next, we use _matcher_ operators to extract sets of _candidate_ spouse relation mentions from the
preprocessed input. We will use these sets as training, development, and test data.

3. **Annotating Evaluation Data [[Intro_Tutorial_3](Intro_Tutorial_3.ipynb)]:**
Next, we use the `Viewer` to label a test set to evaluate against, and/or use helpers to
load external test labels.

4. **Learning [[Intro_Tutorial_4](Intro_Tutorial_4.ipynb)]:**
Here, we go through the process of writing _labeling functions_, learning a generative
model over them, using the generative model to train a _noise-aware_ discriminative
model to make predictions over the candidates, and evaluating the discriminative model
on the development candidate set.

5. **Evaluation [[Intro_Tutorial_5](Intro_Tutorial_5.ipynb)]:**
Finally, we evaluate the learned model on the test candidate set.

## Example

For example, in the sentence (specifically, a photograph caption)
> Prime Minister Lee Hsien Loong and his wife Ho Ching leave a polling station after
> casting their votes in Singapore (Photo: AFP)

our goal is to extract the spouse relation pair ("Lee Hsien Loong", "Ho Ching").

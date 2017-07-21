# Extracting Spouse Relations from the News

In this tutorial, we will walk through the process of using Snorkel to identify mentions
of spouses in a corpus of news articles. The tutorial is broken up into 3 notebooks,
each covering a step in the pipeline:

1. **Preprocessing [[Intro_Tutorial_1](Intro_Tutorial_1.ipynb)]:**
First, we parse the raw input documents into _contexts_ (documents, sentences), and
extract candidate spouse mentions.

2. **Generating _and modeling_ noisy training labels [[Intro_Tutorial_2](Intro_Tutorial_2.ipynb)]:**
Next, we go through the process of writing _labeling functions_ and learning a generative
model to denoise them.

3. **Training an End Extraction Model [[Intro_Tutorial_3](Intro_Tutorial_3.ipynb)]:**
Finally, we train a neural network to identify spouses in the news using our
probabilistic training labels.

## Example

For example, in the sentence (specifically, a photograph caption)
> Prime Minister Lee Hsien Loong and his wife Ho Ching leave a polling station after
> casting their votes in Singapore (Photo: AFP)

our goal is to extract the spouse relation pair ("Lee Hsien Loong", "Ho Ching").

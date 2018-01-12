# Workshop: Rapid Biomedical Knowledge Base Construction from Unstructured Data
---

<img align:right src="imgs/mobilize.jpg" alt="Hazy Research" width=200px/>
<img align:right src="imgs/hazy.jpg" alt="Hazy Research" width=200px/>


This tutorial includes all material presented at the [Mobilize Center's Snorkel Workshop on Rapid Knowledgebase Construction](https://mobilize.stanford.edu/workshop-rapid-biomedical-knowledge-base-construction-from-unstructured-data/)

**Lecture Slides**: [PDF](slides/Snorkel-Workshop-FINAL.pdf)

**Workshop Videos**: [Here](https://simtk.org/frs/?group_id=1263)

### Tutorials

In this tutorial, we will walk through the process of using Snorkel to identify mentions of spouses in a corpus of news articles.

1. **Snorkel API [[Workshop Tutorial 1](Workshop_1_Snorkel_API.ipynb)]:**
We introduct `Candidate` and `Context` objects (documents, sentences) and then show how to interact with candidates using the Snorkel helper function API. 

2. **Writing Labeling Functions [[Workshop Tutorial 2](Workshop_2_Writing_Labeling_Functions.ipynb)]:**
We discuss how to write how to explore our training data, write _labeling functions_, and use _labeling function factories_ to autogenerate LFs from simple dictionaries and regular expressions.

3. **Training the Generative Model [[Workshop Tutorial 3](Workshop_3_Generative_Model_Training.ipynb)]:**
We discuss how to unify the supervision provided by lableing functions in the previous notebook. We show how using a generative model 

4. **Traiing the Discrimintive [[Workshop Tutorial 4](Workshop_4_Discriminative_Model_Training.ipynb)]:**
Using the output of the generative model above, we train a _noise-aware_ discriminative model (here a deep neural network) to make predictions over the candidates, and evaluating the discriminative model on the development candidate set.

### Advanced Tutorials

5. **Preprocessing [[Workshop Tutorial 5](Workshop_5_Advanced_Preprocessing.ipynb)]:**
How to preprocess a corpus of documents and initialize a Snorkel database.

6. **Grid Search [[Workshop Tutorial 6](Workshop_6_Advanced_Grid_Search.ipynb)]:**
Model tuning through grid search.

7. **BRAT Annotator [[Workshop Tutorial 7](Workshop_7_Advanced_BRAT_Annotator.ipynb)]:**
How to construct a validation set of human annotated data using BRAT (Brat Rapid Annotation Tool).

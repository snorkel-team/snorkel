<img align:right src="imgs/acm.jpg" alt="Snorkel" width=700px/>

# Summer School in Software 2.0

---
<img align:right src="imgs/hazy.jpg" alt="Hazy Research" width=275px/>

## What is Software 2.0?

An increasing number of real-world systems today are not just utilizing machine learning as a sub-component, but are in fact wholesale transitioning to “Software 2.0”-style architectures where machine learning models are the principle deployed artifact. Especially for complex tasks e.g. involving vision, speech, control, and more, this approach has the advantages of better generalization as well as a more homogeneous and modular form factor

## Lecture & Workshop Materials

1. **LECTURE: Snorkel Overview** [Slides](slides/Snorkel-Workshop-FINAL.pdf)
2. **INTERACTIVE: Writing Labeling Functions**

	1. **[Snorkel API](Workshop_1_Snorkel_API.ipynb):**
We introduct `Candidate` and `Context` objects (documents, sentences) and then show how to interact with candidates using the Snorkel helper function API. 

	2. **[Writing Labeling Functions] (Workshop_2_Writing_Labeling_Functions.ipynb):**
We discuss how to write how to explore our training data, write _labeling functions_, and use _labeling function factories_ to autogenerate LFs from simple dictionaries and regular expressions.

3. **LECTURE: Weak Supervision Theory**  [Slides](slides/DP_matrix_completion_theory.pdf)

2. **INTERACTIVE: Writing Labeling Functions**
	1. **[Training the Generative Model](Workshop_3_Generative_Model_Training.ipynb):**
	We discuss how to unify the supervision provided by lableing functions in the previous notebook. We show how using a generative model 
	
	2. **[Training the Discrimintive](Workshop_4_Discriminative_Model_Training.ipynb):**
	Using the output of the generative model above, we train a _noise-aware_ discriminative model (here a deep neural network) to make predictions over the candidates, and evaluating the discriminative model on the development candidate set.
	
	3. **[Working with Images](https://github.com/HazyResearch/snorkel/blob/master/tutorials/images/Images_Tutorial.ipynb):**
	Snorkel isn't limited to just text-based classification problems. In this tutorial, we show how Snorkel can be used for computer vision tasks. 

## Advanced Tutorials

These are useful additional tutorials for advanced Snorkel features.

1. **[Data Preprocessing](Workshop_5_Advanced_Preprocessing.ipynb):**
How to preprocess a corpus of documents and initialize a Snorkel database.

2. **[Model Tuning](Workshop_6_Advanced_Grid_Search.ipynb):**
Model tuning through grid search.

3. **[BRAT Annotator](Workshop_7_Advanced_BRAT_Annotator.ipynb):**
How to construct a validation set of human annotated data using BRAT (Brat Rapid Annotation Tool).

## Further Reading on Weak Supervision

If you're new, get started with the first blog post on data programming, and then check out the Snorkel intro tutorial!

1. [Weak Supervision: The New Programming Language for Software 2.0](https://hazyresearch.github.io/snorkel/blog/snorkel_programming_training_data.html)
2. [Systematically Debugging Training Data for Software 2.0](http://dawn.cs.stanford.edu/2018/06/21/debugging/)
3. [Exploiting Building Blocks of Data to Efficiently Create Training Sets](http://dawn.cs.stanford.edu/2017/09/14/coral/)
4. [Weak Supervision: The New Programming Paradigm for Machine Learning](https://hazyresearch.github.io/snorkel/blog/ws_blog_post.html)


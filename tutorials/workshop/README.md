<img align:right src="imgs/acm.jpg" alt="Snorkel" width=700px/>

# Summer School in Snorkel, Weak Supervision & Software 2.0

---
<img align:right src="imgs/hazy.jpg" alt="Hazy Research" width=275px/>

## Programming Software 2.0 with Weak Supervision

In the last few years, deep learning models have simultaneously achieved high quality on conventionally challenging tasks and become easy-to-use commodity tools.	
These factors, combined with the ease of deployment compared to traditional software, have led to deep learning models replacing production software stacks in not only traditional machine learning-driven products including translation and search, but also in many previously heuristic-based applications.
This new mode of software construction and deployment has been called [Software 2.0](https://medium.com/@karpathy).
A key bottleneck in the construction of Software 2.0 applications is the need for large, high-quality training sets for each task.

As labeling training data increasingly becomes one of the most central ways in which developers interact with---and _program_---this new Software 2.0 stack, an emerging area of work focuses on _weak supervision_ techniques for generating labeled training data more efficiently using higher-level, more agile interfaces.
For concreteness, this tutorial focuses on Snorkel, a system that enables users to shape, create, and manage training data for Software 2.0 stacks.
In Snorkel applications, instead of tediously hand-labeling individual data items, a user implicitly defines large training sets by writing programs, called labeling functions, that assign labels to subsets of data points, albeit noisily.
This idea of using multiple, imperfect sources of labels builds on previous work in _distant supervision_, and extends it to handle a more diverse range of noisier, biased, and potentially correlated sources.

In this tutorial, we focus on a basic introduction to the Snorkel paradigm, its interface and workflow, and its motivating context and theory.

## Lecture & Workshop Materials

0. **LECTURE: Software 2.0 Intro** [Slides](slides/ACM_Summer_School_CR.pdf)
1. **LECTURE: Snorkel Overview** [Slides](slides/Snorkel-Workshop-General.pdf)
2. **INTERACTIVE: Writing Labeling Functions**

	1. **[Snorkel API](Workshop_1_Snorkel_API.ipynb):**
We introduct `Candidate` and `Context` objects (documents, sentences) and then show how to interact with candidates using the Snorkel helper function API. 

	2. **[Writing Labeling Functions](Workshop_2_Writing_Labeling_Functions.ipynb):**
We discuss how to write how to explore our training data, write _labeling functions_, and use _labeling function factories_ to autogenerate LFs from simple dictionaries and regular expressions.

3. **LECTURE: Data Programming Theory**  [Slides](slides/DP_matrix_approx_theory.pdf)

2. **INTERACTIVE: Writing Labeling Functions**
	1. **[Training the Generative Model](Workshop_3_Generative_Model_Training.ipynb):**
	We discuss how to unify the supervision provided by lableing functions in the previous notebook. We show how using a generative model 
	
	2. **[Training the Discriminative Model](Workshop_4_Discriminative_Model_Training.ipynb):**
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

Below are some links both on Snorkel and related projects, as well as the broader spectrum of weak supervision work in the community.
For more links, see [the Snorkel home page](snorkel.stanford.edu):

1. [Weak Supervision: The New Programming Language for Software 2.0](https://hazyresearch.github.io/snorkel/blog/snorkel_programming_training_data.html)
2. [Weak Supervision: A Survey Blog Post](https://hazyresearch.github.io/snorkel/blog/ws_blog_post.html)
3. [A Recent NIPS 2017 Workshop on Weak Supervision](http://lld-workshop.github.io/)
4. [Exploiting Building Blocks of Data to Efficiently Create Training Sets](http://dawn.cs.stanford.edu/2017/09/14/coral/)
5. [HoloClean: Data Cleaning using Weak Supervision](https://hazyresearch.github.io/snorkel/blog/holoclean.html)
6. [BabbleLabble: Using Natural Language to Label Training Data](https://hazyresearch.github.io/snorkel/blog/babble_labble.html)
7. [Structure Learning: Handling Correlated Sources Automatically](https://hazyresearch.github.io/snorkel/blog/structure_learning.html), [[Tutorial]](https://github.com/HazyResearch/snorkel/blob/master/tutorials/advanced/Structure_Learning.ipynb)



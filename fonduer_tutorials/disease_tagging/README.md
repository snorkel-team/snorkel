# Disease Tagging Tutorial

In this tutorial, we'll be writing an application to extract **mentions of diseases**
from Pubmed abstracts, as per the
[BioCreative CDR Challenge](http://www.biocreative.org/resources/corpora/biocreative-v-cdr-corpus/).
We will go through the following procedure, with corresponding notebooks:

1. **Preprocessing [[Disease_Tagging_Tutorial_1](Disease_Tagging_Tutorial_1.ipynb)]:**
First we parse the raw input documents into _contexts_ (documents, sentences), and extract
consituent linguistic attributes.

2. **Candidate Extraction [[Disease_Tagging_Tutorial_2](Disease_Tagging_Tutorial_2.ipynb)]:**
Next, we use _matcher_ operators to extract sets of _candidate_ disease mentions from the
preprocessed input. We will use these sets as training, development, and test data.

3. **Creating or Loading Evaluation Labels [[Disease_Tagging_Tutorial_3](Disease_Tagging_Tutorial_3.ipynb)]:**
Next, we use the `Viewer` to label a test set to evaluate against, and/or use helpers to
load external test labels.

4. **Learning [[Disease_Tagging_Tutorial_4](Disease_Tagging_Tutorial_4.ipynb)]:**
Here, we go through the process of writing _labeling functions_, learning a generative
model over them, using the generative model to train a _noise-aware_ discriminative
model to make predictions over the candidates, and evaluating the discriminative model
on the development candidate set.

5. **Evaluation [[Disease_Tagging_Tutorial_5](Disease_Tagging_Tutorial_5.ipynb)]:**
Finally, we evaluate the learned model on the test candidate set.

## Background

This tutorial focuses on the first subtask of the BioCreative's workshop's CDR challenge,
_disease tagging_, which is an important component of information extraction for biomedical
domains such as [medical genetics](http://deepdive.stanford.edu/showcase/apps#genetics)
and [pharmacogenomics](http://deepdive.stanford.edu/showcase/apps#pharmacogenomics).

From the BioCreative workshop's [description](http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/):
> Chemicals, diseases, and their relations are among the most searched topics by PubMed
> users worldwide (1-3) as they play central roles in many areas of biomedical research
> and healthcare such as drug discovery and safety surveillance. Although the ultimate
> goal in drug discovery is to develop chemicals for therapeutics, recognition of adverse
> drug reactions between chemicals and diseases is important for improving chemical safety
> and toxicity studies and facilitating new screening assays for pharmaceutical compound
> survival. In addition, identification of chemicals as biomarkers can be helpful in informing
> potential relationships between chemicals and pathologies. Hence, manual annotation of
> such mechanistic and biomarker/correlative chemical-disease relations (CDR) from
> unstructured free text into structured knowledge has become an important theme for several
> bioinformatics databases such as the Comparative Toxicogenomics Database (CTD) (4). Here
> we consider the words ‘drug’ and ‘chemical’ to be interchangeable.
>
> Manual curation of CDRs from the literature is costly and insufficient to keep up with
> the rapid literature growth. Despite these previous attempts (e.g. (5-7)), free text-based
> automatic biomedical relation detection, from identifying relevant concepts (e.g. diseases
> and chemicals (8-10)) to extracting relations, remains challenging. In addition, few
> relation extraction tools are freely available and to our best knowledge there is limited
> success of using such tools in real-world applications.
>
> An intermediate step for automatic CDR extraction is disease named entity recognition
> and normalization, which was found to be highly difficult on its own (8) in previous
> BioCreative CTD tasks (10,11).
>
> References
>
> 1.	Islamaj Dogan, R., Murray, G.C., Neveol, A., et al. (2009) Understanding PubMed user search behavior through log analysis. Database (Oxford), 2009, bap018.
>
> 2. Lu, Z. (2010) PubMed and beyond: a survey of web tools for searching biomedical literature. Database (Oxford), vol. 2011, baq036.
>
> 3.	Neveol, A., Islamaj Dogan, R., Lu, Z. (2011) Semi-automatic semantic annotation of PubMed queries: a study on quality, efficiency, satisfaction. J Biomed Inform, 44, 310-318.
>
> 4.	Davis, A.P., Grondin, C.J., Lennon-Hopkins, K., et al. (2014) The Comparative Toxicogenomics Database's 10th year anniversary: update 2015. Nucleic Acids Res, 2014 Oct 17,gku935.
>
> 5.	Xu, R., Wang, Q. (2014) Automatic construction of a large-scale and accurate drug-side-effect association knowledge base from biomedical literature. J Biomed Inform.
>
> 6.	Kang, N., Singh, B., Bui, C., et al. (2014) Knowledge-based extraction of adverse drug events from biomedical text. BMC Bioinformatics, 15, 64.
>
> 7.	Gurulingappa, H., Mateen-Rajput, A., Toldo, L. (2012) Extraction of potential adverse drug events from medical case reports. Journal of biomedical semantics, 3, 15.
>
> 8.	Leaman, R., Islamaj Dogan, R., Lu, Z. (2013) DNorm: disease name normalization with pairwise learning to rank. Bioinformatics, 29, 2909-2917.
>
> 9. Leaman, R., Wei, C.H., Lu, Z. (2015) tmChem: a high performance approach for chemical named entity recognition and normalization. Journal of Cheminformatics 2015, 7(Suppl 1):S3
>
> 10.	Wiegers, T.C., Davis, A.P., Mattingly, C.J. (2014) Web services-based text-mining demonstrates broad impacts for interoperability and process simplification. Database (Oxford), 2014, bau050.
>
> 11.	Wiegers, T.C., Davis, A.P., Mattingly, C.J. (2012) Collaborative biocuration--text-mining development task for document prioritization for curation. Database (Oxford), 2012, bas037.

## Example

For example, in the sentence
> The patient had no apparent associated conditions which might have predisposed him to the
> development of bradyarrhythmias; and, thus, this probably represented a true idiosyncrasy
> to lidocaine.

our goal is to tag the disease name "[bradyarrhythmias](https://en.wikipedia.org/wiki/Bradycardia)."


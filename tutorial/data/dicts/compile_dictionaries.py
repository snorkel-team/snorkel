import pandas as pd

from ddbiolib.ontologies.umls import UmlsNoiseAwareDict
from ddbiolib.ontologies.ctd import load_ctd_dictionary
from ddbiolib.ontologies.bioportal import load_bioportal_dictionary

DICT_ROOT = "./"

def get_umls_stopwords(n_max=1,keep_words={}):

    stop_entity_types = ["Quantitative Concept", "Temporal Concept", "Animal", "Food",
                         "Spatial Concept", "Functional Concept"]
    stop_entities = UmlsNoiseAwareDict(positive=stop_entity_types,
                                name="terms", ignore_case=True).dictionary()
    d = {t:1 for t in stop_entities if t not in keep_words and len(t.split()) <= n_max}
    return d

# ==================================================================
# Prefixes & Suffixes
# ==================================================================
# manually created
inheritance = ["x linked", "x linked recessive", "x linked dominant",
               "x-linked", "x-linked recessive", "x-linked dominant",
               "recessive", "dominant", "semidominant", "non-familial",
               "inherited", "hereditary", "nonhereditary", "familial",
               "autosomal recessive", "autosomal dominant"]

# ==================================================================
# The UMLS Semantic Network
# ==================================================================
# The UMLS defines 133 fine-grained entity types which are them
# grouped into coarser semantic categories using groupings defined at:
#    https://semanticnetwork.nlm.nih.gov/download/SemGroups.txt
# This set represents semantic types corresponding to "Disorders"
# with the sub-entity "Finding" removed due to precision issues (10pt drop!)

disease_entity_types = ["Acquired Abnormality",
                        "Anatomical Abnormality",
                        "Cell or Molecular Dysfunction",
                        "Congenital Abnormality",
                        "Disease or Syndrome",
                        "Experimental Model of Disease",
                        "Injury or Poisoning",
                        "Mental or Behavioral Dysfunction",
                        "Neoplastic Process",
                        "Pathologic Function",
                        "Sign or Symptom"]

# UMLS terms and abbreviations/acronyms
umls_disease_terms = UmlsNoiseAwareDict(positive=disease_entity_types, rm_sab=["LPN"], name="terms", ignore_case=False)
# Disease Abbreviations / Acronyms
umls_disease_abbrvs = UmlsNoiseAwareDict(positive=disease_entity_types, rm_sab=["LPN"], name="abbrvs", ignore_case=False)

stop_entity_types = ["Geographic Area",
                     "Genetic Function"]
umls_stop_terms = UmlsNoiseAwareDict(positive=stop_entity_types, name="terms", ignore_case=True)

# ==================================================================
# The National Center for Biomedical Ontology
# http://bioportal.bioontology.org/
#
# Comparative Toxicogenomics Database
# http://ctdbase.org/
# ==================================================================
# This uses 4 disease-related ontologies:
#   (ordo) Orphanet Rare Disease Ontology
#   (doid) Human Disease Ontology
#   (hp)   Human Phenotype Ontology
#   (ctd)  Comparative Toxicogenomics Database

dict_ordo = load_bioportal_dictionary("{}ordo.csv".format(DICT_ROOT))
dict_doid = load_bioportal_dictionary("{}DOID.csv".format(DICT_ROOT))
dict_hp   = load_bioportal_dictionary("{}HP.csv".format(DICT_ROOT))
dict_ctd  = load_ctd_dictionary("{}CTD_diseases.tsv".format(DICT_ROOT))

# ==================================================================
# Manually Created Dictionaries
# ==================================================================
# The goal is to minimize this part as much as possible
# IDEALLY we should build these from the above external curated resources
# Otherwise these are put together using Wikipedia and training set debugging

# Common disease acronyms
fname = "{}common_disease_acronyms.txt".format(DICT_ROOT)
dict_common_disease_acronyms = dict.fromkeys([l.strip() for l in open(fname,"rU")])

fname = "{}stopwords.txt".format(DICT_ROOT)
dict_stopwords = dict.fromkeys([l.strip() for l in open(fname,"rU")])

fname = "{}manual_stopwords.txt".format(DICT_ROOT)
dict_common_stopwords = dict.fromkeys([l.strip() for l in open(fname,"rU")])

# not disease name for this tasks
dict_common_stopwords.update(dict.fromkeys(["disease", "diseases", "syndrome", "syndromes", "disorder",
                                            "disorders", "damage", "infection", "bleeding", "injury"]))


diseases = umls_disease_terms.dictionary(min_size=50)
abbrvs = umls_disease_abbrvs.dictionary(min_size=50)
stop_entities = umls_stop_terms.dictionary(min_size=50)

diseases.update(dict_ordo)
diseases.update(dict_doid)
diseases.update(dict_hp)
diseases.update(dict_ctd)
diseases.update(abbrvs)

#
# Abbreviations/Acronymns
#
# FIX UPPERCASE ISSUE WITH NON-ACRONYMS
# Update with all uppercase terms from diseases
abbrvs.update({term:1 for term in diseases if term.isupper() and len(term) > 1})

dict_common_disease_acronyms.update(dict.fromkeys(["TNSs", "LIDs", "TDP", "TMA", "TG", "SE", "ALF", "CHC", "RPN", "HITT",
                                                   "VTE", "HEM", "NIN", "LID", "CIMD", "MAHA", "LID", "CIMD", "MAHA"]))
abbrvs.update(dict.fromkeys(dict_common_disease_acronyms))


# remove stopwords and stop entity types
stopwords = dict_stopwords
stopwords.update({term.lower():1 for term in stop_entities})
stopwords.update(dict.fromkeys(dict_common_stopwords))

disease_or_syndrome = umls_disease_terms.dictionary(semantic_types=["disease_or_syndrome"])
stopwords.update(get_umls_stopwords(keep_words=disease_or_syndrome))

stopwords.update(dict.fromkeys(inheritance))

diseases = {t.lower().strip():1 for t in diseases if t.lower().strip() not in stopwords and len(t) > 1}

diseases = {t.lower().strip():1 for t in diseases if not t.lower().strip().startswith('heparin-induced')}
abbrvs = {t:1 for t in abbrvs if len(t) > 1 and t not in stopwords}

# we have these in the UMLS -- why do they get stripped out?
diseases.update(dict.fromkeys(["melanoma", "qt prolongation", "seizure", "overdose", "tdp"]))

# general disease
diseases.update(dict.fromkeys(["pain", "hypertension", "hypertensive", "depression", "depressive", "depressed",
                               "bleeding", "infection", "poisoning", "anxiety", "deaths", "startle"]))

# common disease
diseases.update(dict.fromkeys(['parkinsonian', 'convulsive', 'leukocyturia', 'bipolar', 'pseudolithiasis',
                               'malformations', 'angina', 'dysrhythmias', 'calcification', 'paranoid', 'hiv-infected']))

# adj disease
diseases.update(dict.fromkeys(['acromegalic', 'akinetic', 'allergic', 'arrhythmic', 'arteriopathic', 'asthmatic',
                               'atherosclerotic', 'bradycardic', 'cardiotoxic', 'cataleptic', 'cholestatic',
                               'cirrhotic', 'diabetic', 'dyskinetic', 'dystonic', 'eosinophilic', 'epileptic',
                               'exencephalic', 'haemorrhagic', 'hemolytic', 'hemorrhagic', 'hemosiderotic', 'hepatotoxic'
                               'hyperalgesic', 'hyperammonemic', 'hypercalcemic', 'hypercapnic', 'hyperemic',
                               'hyperkinetic', 'hypertrophic', 'hypomanic', 'hypothermic', 'ischaemic', 'ischemic',
                               'leukemic', 'myelodysplastic', 'myopathic', 'necrotic', 'nephrotic', 'nephrotoxic',
                               'neuropathic', 'neurotoxic', 'neutropenic', 'ototoxic', 'polyuric', 'proteinuric',
                               'psoriatic', 'psychiatric', 'psychotic', 'quadriplegic', 'schizophrenic', 'teratogenic',
                               'thromboembolic', 'thrombotic', 'traumatic', 'vasculitic']))

# remove disease name like chronic renal failure (CRF)
for d in diseases.keys():
    if d[-1]==')':
        s=d.split(' (')
        if len(s)==2:
            s1,s2=s[0],s[1][:-1]
            s3=''.join([i[0] for i in s1.replace('-', ' ').split()]).lower()
            if s2.lower()==s3.lower():
                del diseases[d]
                diseases[s1]=1
                abbrvs[s2.upper()]=1
            s3=''.join([i[0] for i in s1.split()]).lower()
            if s2.lower()==s3.lower() and d in diseases:
                del diseases[d]
                diseases[s1]=1
                abbrvs[s2.upper()]=1

# remove disease name contains general term
for d in diseases.keys():
    if d.lower().split()[0] in ['generalized', 'metastatic', 'recurrent', 'complete', 'immune']:
        del diseases[d]

diseases = pd.Series(diseases.keys())
diseases.to_csv(DICT_ROOT + 'disease_names.csv', encoding='utf-8')

abbrvs = pd.Series(abbrvs.keys())
abbrvs.to_csv(DICT_ROOT + 'disease_abbrvs.csv', encoding='utf-8')

# #
# # DISEASE WITH BODY PART
# #
# #
body_part = UmlsNoiseAwareDict(positive=["Body Part, Organ, or Organ Component", "Body Location or Region"],
                                name="*", ignore_case=True).dictionary()

specical_body_part = ['back']

functional_concept = {t.lower():1 for t in body_part if len(t) > 1 and
                      t.lower() not in stopwords and not t.isdigit()}

disease_pattern = ["disease", "diseases", "syndrome", "syndromes", "disorder", "disorders", "damage", "infection",
       "lesion", "lesions", "impairment", "impairments", "failure", "failures", "occlusion", "occlusions",
       "dysfunction", "dysfunctions", "toxicity", "injury", "carcinoma", "carcinomas", "thrombosis", "cancer",
       "cancers", "block", "pain"]

timestamp = ["end-stage", "acute", "chronic", "congestive"]

conjunction = ["and", "or", "and/or"]

body_part.update(functional_concept)

body_part = pd.Series(body_part.keys())
body_part.to_csv(DICT_ROOT + 'body_parts.csv', encoding='utf-8')

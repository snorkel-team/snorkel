import re
from lf_terms import *
from snorkel.lf_helpers import *
from utils import *


# Load dictionaries
diseases = load_disease_dictionary()
diseases.update(load_acronym_dictionary())


###
### More mention level LFs:
###

def LF_common_disease_acronyms(c):
    '''Common disease acronyms'''
    return 1 if " ".join(c[0].get_attrib_tokens()) in common_disease_acronyms else 0


def LF_deficiency_of(c):
    '''deficiency of <TYPE>'''
    phrase = " ".join(c[0].get_attrib_tokens()).lower()
    return 1 if phrase.endswith('deficiency') or phrase.startswith('deficiency') or phrase.endswith('dysfunction') else 0


def LF_positive_indicator(c):
    flag = False
    for i in c[0].get_attrib_tokens():
        if i.lower() in positive_indicator:
            flag = True
            break
    return 1 if flag else 0


def LF_left_positive_argument(c):    
    phrase = " ".join(c[0].get_attrib_tokens('lemmas')).lower()
    pattern = "(\w+ ){1,2}(infection|lesion|neoplasm|attack|defect|anomaly|abnormality|degeneration|carcinoma|lymphoma|tumor|tumour|deficiency|malignancy|hypoplasia|disorder|deafness|weakness|condition|dysfunction|dystrophy)$"
    return 1 if re.search(pattern,phrase) else 0


def LF_right_negative_argument(c):    
    phrase = " ".join(c[0].get_attrib_tokens('lemmas')).lower()
    pattern = "^(history of|mitochondrial|amino acid)( \w+){1,2}"
    return 1 if re.search(pattern,phrase) else 0


def LF_medical_afixes(c):
    pattern = "(\w+(pathy|stasis|trophy|plasia|itis|osis|oma|asis|asia)$|^(hyper|hypo)\w+)"
    phrase = " ".join(c[0].get_attrib_tokens('lemmas')).lower()
    return 1 if re.search(pattern,phrase) else 0


def LF_adj_diseases(c):
    return 1 if ' '.join(c[0].get_attrib_tokens()) in adj_diseases else 0


###
### More negative LFs:
###

def LF_too_vague(c):
    phrase = " ".join(c[0].get_attrib_tokens('lemmas')).lower()
    phrase_ = " ".join(c[0].get_attrib_tokens()).lower()
    return -1 if phrase in vague or phrase_ in vague else 0


def LF_neg_surfix(c):
    terms = ['deficiency', 'the', 'the', 'of', 'to', 'a']
    rw = get_right_tokens(c, window=1, attrib='lemmas')
    if len(rw) > 0 and rw[0].lower() in terms:
        return -1
    return 0


def LF_non_common_disease(c):
    '''Non common diseases'''
    return -1 if " ".join(c[0].get_attrib_tokens()).lower() in non_common_disease else 0


def LF_non_disease_acronyms(c):
    '''Non common disease acronyms'''
    return -1 if " ".join(c[0].get_attrib_tokens()) in non_disease_acronyms else 0


def LF_pos_in(c):
    '''Candidates beginning with a preposition or subordinating conjunction'''
    pos_tags = c[0].get_attrib_tokens('pos_tags')
    return -1 if "IN" in pos_tags[0:1] else 0


def LF_gene_chromosome_link(c):
    '''Mentions of the form "Huntington Disease gene"'''
    genetics_terms = set(["gene","chromosome"])
    diseases_terms = set(["disease","syndrome","disorder"])
    context = get_left_tokens(c,window=10, attrib='lemmas') + get_right_tokens(c,window=10, attrib='lemmas')

    # 1: contains a disease keyword or 2: in disease dictionaries
    is_disease = diseases_terms.intersection(map(lambda x:x.lower(), c[0].get_attrib_tokens()))
    is_disease = is_disease or " ".join(c[0].get_attrib_tokens()) in diseases
    is_gene = genetics_terms.intersection(context)    
    return -1 if is_gene and not is_disease else 0


def LF_right_window_incomplete(c):
    return -1 if right_terms.intersection(get_right_tokens(c,window=2, attrib='lemmas')) else 0


def LF_negative_indicator(c):
    flag = False
    for i in c[0].get_attrib_tokens():
        if i.lower() in negative_indicator:
            flag = True
            break
    return -1 if flag else 0

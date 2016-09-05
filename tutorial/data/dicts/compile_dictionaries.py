import bz2
import csv
import os
import re
import sys
import codecs
import cPickle
import itertools
import numpy as np

# NOTE: This requires the ddbiolib repo: https://github.com/HazyResearch/ddbiolib
from ddbiolib.datasets import cdr
from ddbiolib.ontologies.umls import UmlsNoiseAwareDict
from ddbiolib.ontologies.ctd import load_ctd_dictionary
from ddbiolib.ontologies.specialist import SpecialistLexicon
from ddbiolib.ontologies.bioportal import load_bioportal_dictionary
from ddbiolib.utils import unescape_penn_treebank

ROOT = os.path.join(os.environ['SNORKELHOME'], 'tutorial/data/dicts/')
print "Using root=", ROOT

def load_bioportal_csv_dictionary(filename):
    """Load BioPortal Ontologies--http://bioportal.bioontology.org/"""
    reader = csv.reader(open(filename,"rU"),delimiter=',', quotechar='"')
    d_in = [line for line in reader]
    d = []
    for line in d_in[1:]:
        row = dict(zip(d_in[0],line))
        d.append(row["Preferred Label"])
        d += row["Synonyms"].split("|")
    return d
    
def load_disease_dictionary():  
    """
    Load a dictionary of disease phrases **as a list**.
    NOTE: Eventually we'll want to pass along IDs
    """
    d = set()
      
    # UMLS SemGroup Disorders
    # TODO: Re-evaluate how to use this... too many general terms??
    dictfile = ROOT + "/umls_disorders_v2.bz2"
    diseases = [line.strip().split("\t")[0] for line in bz2.BZ2File(dictfile, 'rb').readlines()]
    d.update(w for w in diseases if not w.isupper())

    # Orphanet Rare Disease Ontology
    d.update(w for w in load_bioportal_csv_dictionary(ROOT + "/ordo.csv") if not w.isupper())
    
    # Human Disease Ontology 
    d.update(w for w in load_bioportal_csv_dictionary(ROOT + "/DOID.csv") if not w.isupper())
      
    # ------------------------------------------------------------
    # remove cell dysfunction terms
    dictfile = ROOT + "/cell_molecular_dysfunction.txt"
    remove_terms = set(line.strip().split("\t")[0] for line in open(dictfile).readlines())

    # remove geographic areas terms
    dictfile = ROOT + "/umls_geographic_areas.txt"
    remove_terms.update(line.strip().split("\t")[0] for line in open(dictfile).readlines())
    d = d.difference(remove_terms)
    
    # remove stopwords
    dictfile = ROOT + "/stopwords.txt"
    stopwords = [line.strip().split("\t")[0] for line in open(dictfile).readlines()]
    d = [w for w in list(d) if w.lower() not in stopwords and len(w) > 0]

    # remove a manually-created stopwords dictionary based on looking at most-frequent unary terms
    # TODO: These are all from a subtree of the UMLS that should be dropped there, this is just a temporary
    # workaround until we do that!!!
    dictfile = ROOT + "/stopwords_most_frequent.tsv"
    stopwords = [line.strip().split("\t")[0] for line in open(dictfile).readlines()]
    d = [w for w in list(d) if w.lower() not in stopwords and len(w) > 0]
    return d


def load_disease_acronym_dictionary():    
    """
    Load a dictionary of disease phrases **as a list**.
    NOTE: Eventually we'll want to pass along IDs
    """
    a = set()
    
    # UMLS disorders
    dictfile = ROOT + "/umls_disorders_v2.bz2"
    diseases = [line.strip().split("\t")[0] for line in bz2.BZ2File(dictfile, 'rb').readlines()]
    a.update(w for w in diseases if w.isupper())
    
    # Orphanet Rare Disease Ontology
    a.update(w for w in load_bioportal_csv_dictionary(ROOT + "/ordo.csv") if w.isupper())
    
    # Human Disease Ontology 
    a.update(w for w in load_bioportal_csv_dictionary(ROOT + "/DOID.csv") if w.isupper())
    
    # filter by char length
    a = [w for w in list(a) if len(w) > 1]
    return a


def load_chemicals_dictionary():
    """Load a dictionary of chemical phrases as a list."""
    entity_types = [
        'Antibiotic', 'Carbohydrate', 'Chemical', 'Eicosanoid', 'Element, Ion, or Isotope',
        'Hazardous or Poisonous Substance', 'Indicator, Reagent, or Diagnostic Aid', 'Inorganic Chemical',
        'Neuroreactive Substance or Biogenic Amine', 'Nucleic Acid, Nucleoside, or Nucleotide', 
        'Organic Chemical', 'Organophosphorus Compound', 'Steroid', 'Vitamin', 'Lipid']

    umls_terms = UmlsNoiseAwareDict(positive=entity_types, name="terms", ignore_case=False)
    umls_abbrv = UmlsNoiseAwareDict(positive=entity_types, name="abbrvs", ignore_case=False)

    chemicals = umls_terms.dictionary()
    acronyms = umls_abbrv.dictionary()

    # remove stopwords
    fname = ROOT + "/chem_stopwords.txt"
    stopwords = dict.fromkeys([line.strip().split("\t")[0] for line in open(fname).readlines()])
    stopwords.update(dict.fromkeys(["V","IV","III","II","I","cm","mg","pH","In", "Hg", "VIP"]))
    diseases = [
        "pain", "hypertension", "hypertensive", "depression", "depressive", "depressed", "bleeding", "infection", 
        "poisoning", "anxiety", "deaths", "startle"]
    vague = [
        "drug", "drugs", "control", "animals", "animal", "related", "injection", "level", "stress", "baseline",
        "oral"]

    # From error analysis
    vague += [
        "placebo", "hepatitis", "mediated", "therapeutic", "purpose", "block", "various", "active", "medication", 
        "dopaminergic", "prevent", "blockade", "conclude", "mouse", "acid", "support", "medications", "lipid",
        "lipids", "prolactin", "neuronal", "central nervous system", "water", "tonic", "task", "basis", "topical",
        "hemoglobin", "diagnostic", "pressor", "compound", "solution", "hg", "nervous system", "hepatitis b",
        "analgesic", "triad", "anti-inflammatory", "opioid", "metabolites", "adrenoceptor", "immunosuppressive",
        "prophylactic", "unknown", "antioxidant", "anticonvulsant", "inhibitors", "food", "anesthetic",
        "antiarrhythmic", "retinal", "complex", "antibody", "combinations", "antiepileptic", "component",
        "contrast", "stopping", "chemical", "label", "sham", "salt", "transcript", "s-1", "glucocorticoid",
        "glucocorticoids"]

    stopwords.update(dict.fromkeys(diseases))
    stopwords.update(dict.fromkeys(vague))

    chemicals = {t.lower().strip():1 for t in chemicals if t.lower().strip() not in [i.lower() for i in stopwords.keys()] and len(t) > 1}
    acronyms = {t.strip():1 for t in acronyms if t.strip() not in stopwords and len(t) > 1}
    ban = [
        "antagonist", "receptor", "agonist", "transporter", "channel", "monohydrate", "phosphokinase", "kinase"]
    ban += [
        "drug", "control", "related", "animals", "injection", "level", "duration", "baseline", "agent", "stress", 
        "liposomal", "vehicle", "total", "pain"]

    # filter out some noisy dictionary matches
    for phrase in chemicals.keys():
        check=False
        for i in ban:
            if i.lower() in phrase.lower():
                check=True
                break
        if phrase.endswith('ic'):
            check=True
        if phrase.endswith('+'):
            check=True
        a=phrase.lower().split()
        if len(a)==2 and a[0].isdigit() and a[1]=='h':
            check=True
        if check:
            del chemicals[phrase]

    acronyms.update(dict.fromkeys([
        "CaCl(2)", "PAN", "SNP", "K", "AX", "VPA", "PG-9", "SRL", "ISO", "CAA", "CBZ", "CPA", 
        "GEM", "CY", "OC", "Ca", "PTZ", "NMDA", "H2O", "CsA", "DA", "GSH", "HBsAg", "Rg1"]))

    chemicals.update(dict.fromkeys([
        "glutamate", "aspartate", "creatine", "angiotensin", "glutathione", "srl", "dex", "tac",
        "cya", "l-dopa", "hbeag", "argatroban", "melphalan", "cyclosporine", "enalapril", 
        "l-arginine", "vasopressin", "cyclosporin a", "n-methyl-d-aspartate", "ace inhibitor", 
        "oral contraceptives", "l-name", "alanine", "amino acid", "lisinopril", "tyrosine", 
        "fenfluramines", "beta-carboline", "glutamine", "octreotide", "antidepressant"]))
    
    # Filter empty / single-char entries + convert to list
    chemicals = [c for c in chemicals.keys() if len(c) > 1]
    acronyms  = [a for a in acronyms.keys() if len(a) > 1]

    # Filter integers
    chemicals = filter(lambda x : not x.isdigit(), chemicals)
    acronyms  = filter(lambda x : not x.isdigit(), acronyms)

    return chemicals, acronyms


if __name__ == '__main__':

    # Save diseases dictionary
    print 'Compiling disease dictionary...'
    disease_phrases = load_disease_dictionary()
    print 'Loaded:', len(disease_phrases)
    open(ROOT + '/disease_phrases.txt', 'w+').write('\n'.join(disease_phrases))

    # Save disease acronyms dictionary
    print 'Compiling disease acronyms...'
    disease_acronyms = load_disease_acronym_dictionary()
    print 'Loaded:', len(disease_acronyms)
    open(ROOT + '/disease_acronyms.txt', 'w+').write('\n'.join(disease_acronyms))

    # Save chemical and chemical acronyms dictionaries
    print 'Compiling chemical dictionaries...'
    chemical_phrases, chemical_acronyms = load_chemicals_dictionary()
    print 'Loaded:', len(chemical_phrases)
    with open(ROOT + '/chemical_phrases.txt', 'w+') as f:
        for p in chemical_phrases:
            try:
                f.write(p + '\n')
            except UnicodeEncodeError:
                pass
    print 'Loaded:', len(chemical_acronyms)
    with open(ROOT + '/chemical_acronyms.txt', 'w+') as f:
        for p in chemical_acronyms:
            try:
                f.write(p + '\n')
            except UnicodeEncodeError:
                pass

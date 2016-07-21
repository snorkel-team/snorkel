import bz2
import csv
import os

ROOT = os.path.join(os.environ['SNORKELHOME'], 'tutorial/data/dicts/')

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
    d = [w for w in list(d) if w.lower() not in stopwords]
    return d

def load_acronym_dictionary():    
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

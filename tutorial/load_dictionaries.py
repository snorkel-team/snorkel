import bz2
import csv

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
      
    # UMLS SemGroup Disorders
    dictfile = "data/dicts/umls_disorders_v2.bz2"
    diseases = [line.strip().split("\t")[0] for line in bz2.BZ2File(dictfile, 'rb').readlines()]
    diseases = [word for word in diseases if not word.isupper()]

    # Orphanet Rare Disease Ontology
    ordo = load_bioportal_csv_dictionary("data/dicts/ordo.csv")
    ordo = {word:1 for word in ordo if not word.isupper()}
    diseases.update(ordo)
    
    # Human Disease Ontology 
    doid = load_bioportal_csv_dictionary("data/dicts/DOID.csv")
    doid = {word:1 for word in doid if not word.isupper()}
    diseases.update(doid)
      
    # ------------------------------------------------------------
    # remove cell dysfunction terms
    dictfile = "data/dicts/cell_molecular_dysfunction.txt"
    terms = [line.strip().split("\t")[0] for line in open(dictfile).readlines()]
    diseases = {word:1 for word in diseases if word not in terms} 
    
    dictfile = "data/dicts/umls_geographic_areas.txt"
    terms = [line.strip().split("\t")[0] for line in open(dictfile).readlines()]
    diseases = {word:1 for word in diseases if word not in terms}
    # ------------------------------------------------------------
    
    # NCBI training set vocabulary
    dictfile = "data/dicts/ncbi_training_diseases.txt"
    terms = [line.strip().split("\t")[0] for line in open(dictfile).readlines()]
    terms = {word:1 for word in terms if not word.isupper()}
    diseases.update(terms)
    
    # remove stopwords
    dictfile = "data/dicts/stopwords.txt"
    stopwords = [line.strip().split("\t")[0] for line in open(dictfile).readlines()]
    diseases = {word:1 for word in diseases if word.lower() not in stopwords}  
    
    return diseases

def load_acronym_dictionary():    
    dictfile = "data/dicts/umls_disorders_v2.bz2"
    diseases = {line.strip().split("\t")[0]:1 for line in bz2.BZ2File(dictfile, 'rb').readlines()}
    diseases = {word:1 for word in diseases if word.isupper()}
    
    # Orphanet Rare Disease Ontology
    ordo = load_bioportal_csv_dictionary("data/dicts/ordo.csv")
    ordo = {word:1 for word in ordo if word.isupper()}
    diseases.update(ordo)
    
    # Human Disease Ontology 
    doid = load_bioportal_csv_dictionary("data/dicts/DOID.csv")
    doid = {word:1 for word in doid if word.isupper()}
    diseases.update(doid)
    
    dictfile = "data/dicts/ncbi_training_diseases.txt"
    terms = [line.strip().split("\t")[0] for line in open(dictfile).readlines()]
    terms = {word:1 for word in terms if word.isupper()}
    diseases.update(terms)
    
    # filter by char length
    diseases = {word:1 for word in diseases if len(word) > 1}
    
    return diseases

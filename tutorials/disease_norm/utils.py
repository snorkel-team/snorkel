import lxml.etree as et
from snorkel.models import CandidateSet, split_stable_id
from snorkel.candidates import TemporarySpan
from collections import defaultdict, namedtuple
import os
import codecs
import csv
import bz2
import re
import numpy as np
from itertools import permutations
from scipy.sparse import lil_matrix


class CanonicalDictionary(object):
    """Class for loading, pre-processing and storing the (merged) canonical dictionary for an entity type"""
    def __init__(self, sid_to_cid_map):
        """
        Takes as input a pre-computed map from source ID (sid) to canonical ID (cid)
        Any source ID not in this mapping is assumed to *not* be of the correct entity class, i.e. cid = -1
        *However*, we still store the tree path of these non-entity SIDs for use in e.g. negative LFs
        """
        
        # Take in a pre-computed mapping from canonical IDs to integer (> 0) ids to be used by system
        # Also construct inverse map
        self.sid_to_cid    = sid_to_cid_map
        self.cid_to_sid    = {}
        for sid, cid in sid_to_cid_map.iteritems():
            self.cid_to_sid[cid] = sid

        # Mapping from input terms to CID
        self.term_to_sids = defaultdict(set)

        # Also maintain an (ordered) list of terms, which will correspond to rows of a CD matrix
        self.terms = []

        # Mapping from CIDs to ontology trees (provided as lists of node IDs from the root)
        self.tree_paths = defaultdict(list)

    def add_term(self, term, sid, tree_paths=[]):
        """Add a term, id, and *list of* tree paths (each one of which is a list)"""
        terms = self._process_term(term)
        for t in terms:
            if len(self.term_to_sids[t]) == 0:
                self.terms.append(t)
            self.term_to_sids[t].add(sid)
        
        for tp in tree_paths:
            if tp not in self.tree_paths[sid]:
                self.tree_paths[sid].append(tp)

    def _process_term(self, t, min_len=3):
        """Takes in a raw string, returns a set of string terms"""
        out = set()

        # Lower-case
        t = t.lower()

        # Min-length thresholding
        if len(t) < min_len:
            return out

        # Handle (x) entries?
        # TODO

        # Handle entries ending in r'(group )?\d'?
        # TODO

        # Transform comma-style entries
        if ',' in t:
            splits = t.split(',')
            out.add(" ".join(s.strip() for s in splits[1:]) + " " + splits[0].strip())
            out.add(" ".join(s.strip() for s in splits[1:][::-1]) + " " + splits[0].strip())
        else:
            out.add(t)
        return out


MEDICEntry = namedtuple('MEDICEntry', 'name, id, defn, alt_ids, parent_ids, tree_nums, parent_tree_nums, synonyms, categories')

def split_pipe_delim(p):
    if len(p) == 0:
        return []
    else:
        return p.split('|')

def load_MEDIC(filepath='data/CTD_diseases.csv'):
    """Loads a list of MEDICEntry rows, and a mapping from MEDIC ids -> integer CIDS"""
    MEDIC_to_CID = {}

    # Note: need to delete the first 29 comment rows in the raw source file first!
    medic_entries = []
    with open(filepath, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:

            # Add to MEDIC -> CID mapping, leaving 0 reserved
            if row[1] not in MEDIC_to_CID:
                MEDIC_to_CID[row[1]] = len(MEDIC_to_CID) + 1
                                                
            entry = MEDICEntry(
                name             = row[0],
                id               = row[1],
                defn             = row[2],
                alt_ids          = split_pipe_delim(row[3]),
                parent_ids       = split_pipe_delim(row[4]),
                tree_nums        = split_pipe_delim(row[5]),
                parent_tree_nums = split_pipe_delim(row[6]),
                synonyms         = split_pipe_delim(row[7]),
                categories       = split_pipe_delim(row[8])
            )
            medic_entries.append(entry)
    
    print "Loaded %s MEDIC entries" % len(medic_entries)
    return medic_entries, MEDIC_to_CID


def binarize_LF_matrix(X):
    X_b = lil_matrix(X.shape)
    for i, j in zip(*X.nonzero()):
        X_b[i,j] = np.sign(X[i,j])
    return X_b.tocsr()


def get_binarized_score(predicted, gold):
    tp = 0
    pp = 0
    p  = 0
    for i in range(gold.shape[0]):
        if gold[i] > 0:
            p += 1
                    
        if predicted[i] == 1:
            pp += 1
            if gold[i] > 0:
                tp += 1
                    
    prec   = tp / float(pp)
    recall = tp / float(p)
    f1     = (2*prec*recall) / (prec+recall)
    print "P :\t", prec
    print "R :\t", recall
    print "F1:\t", f1


def get_docs_xml(filepath, doc_path=".//document", id_path=".//id/text()"):
    xml = et.fromstring(open(filepath, 'rb').read())
    return dict(zip(xml.xpath(id_path), xml.xpath(doc_path)))


def get_CD_mentions_by_MESHID(doc_xml, sents):
    """
    Collect a set of Pubtator chemical-induced disease (CID) relation mention annotations.
    Returns a dictionary of (sent_id, char_start, char_end) tuples indexed by MESH ID.
    """
    # We get the sentence offsets *relative to document start* by unpacking their stable ids
    sent_offsets = [split_stable_id(s.stable_id)[2] for s in sents]

    # Get unary mentions of diseases / chemicals
    unary_mentions = defaultdict(lambda : defaultdict(list))
    annotations = doc_xml.xpath('.//annotation')
    for a in annotations:

        # NOTE: Ignore CompositeRole individual mention annotations for now
        comp_roles = a.xpath('./infon[@key="CompositeRole"]/text()')
        comp_role = comp_roles[0] if len(comp_roles) > 0 else None
        if comp_role == 'IndividualMention':
            continue

        # Get basic annotation attributes
        txt = a.xpath('./text/text()')[0]
        offset = int(a.xpath('./location/@offset')[0])
        length = int(a.xpath('./location/@length')[0])
        type = a.xpath('./infon[@key="type"]/text()')[0]
        mesh = a.xpath('./infon[@key="MESH"]/text()')[0]
        
        # Get sentence id and relative character offset
        si = len(sent_offsets) - 1
        for i,so in enumerate(sent_offsets):
            if offset == so:
                si = i
                break
            elif offset < so:
                si = i - 1
                break
        sent       = sents[si]       
        char_start = offset - sent_offsets[si]
        char_end   = char_start + length - 1
        
        # Index by MESH ID as that is how relations refer to
        unary_mentions[type][mesh].append((sent, char_start, char_end, txt))
    return unary_mentions


def get_CID_unary_mentions(doc_xml, doc, type):
    """
    Get a set of unary disease mentions in argument-dict format,
    for ExternalAnnotationsLoader
    """
    for mesh_id, ms in get_CD_mentions_by_MESHID(doc_xml, doc.sentences)[type].iteritems():
        for m in ms:
            yield {type.lower() : TemporarySpan(parent=m[0], char_start=m[1], char_end=m[2])}


def get_CID_relations(doc_xml, doc):
    """
    Given the doc XML and extracted unary mention tuples, return pairs of unary mentions that are annotated
    as CID relations.

    NOTE: This is somewhat ambiguous as relations are only marked at the entity level here...
    """
    unary_mentions = get_CD_mentions_by_MESHID(doc_xml, doc.sentences)
    annotations    = doc_xml.xpath('.//relation')
    for a in annotations:
        cid = a.xpath('./infon[@key="Chemical"]/text()')[0]
        did = a.xpath('./infon[@key="Disease"]/text()')[0]
        for chemical in unary_mentions['Chemical'][cid]:
            for disease in unary_mentions['Disease'][did]:

                # Only take relations in same sentence
                if chemical[0] == disease[0]:
                    yield (chemical, disease)


def load_mesh_raw(fname):
    """Loads full MESH tree as simple list of (MESH ID, [Tree #s], [terms]) entries"""
    root    = et.parse(fname).getroot()
    entries = []

    for d in root:
        mesh_id    = d.find('DescriptorUI').text
        term_names = []

        # Put into dictionary if part of any tree with the correct prefix
        tree_nums = []
        trees     = d.find('TreeNumberList')
        if trees is not None:
            tree_nums = [t.text for t in trees]

        # Get string entries
        term_names.append(d.find('DescriptorName').find('String').text.lower())
        concepts = d.find('ConceptList')
        for c in concepts:
            terms = c.find('TermList')
            if terms is not None:
                term_names.extend(term.find('String').text.lower() for term in terms)

        entries.append((mesh_id, tree_nums, term_names))
    print "Loaded %s entries" % len(entries)
    return entries


def load_mesh_dict(fname, tree_prefixes=[]):
    """
    Loads a dictionary mapping lower-case terms -> MESH IDs from the mesh descYYYY.xml file
    E.g. data/desc2017.xml
    Optionally filters to certain part of the MESH concept tree by matching tree prefix
    (ex: "C" for diseases, "D" for chemicals and drugs, etc.)
    """
    root = et.parse(fname).getroot()
    print "Loaded XML"
    mesh_dict = {}
    for d in root:
        mesh_id = d.find('DescriptorUI').text
        name = d.find('DescriptorName').find('String').text

        # Put into dictionary if part of any tree with the correct prefix
        if len(tree_prefixes) > 0:
            trees = d.find('TreeNumberList')
            if trees is None:
                continue
            is_relevant = False
            for tree in trees:
                if any([tree.text.startswith(p) for p in tree_prefixes]):
                    is_relevant = True
                    break
            if not is_relevant:
                continue

        # Get string entries
        concepts = d.find('ConceptList')
        term_names = [name]
        for c in concepts:
            terms = c.find('TermList')
            if terms is not None:
                term_names.extend(term.find('String').text for term in terms)

        # Add lower cased
        for t in term_names:
            mesh_dict[t.lower()] = mesh_id
    return mesh_dict


#####################################################################
# DICTIONARY LOADERS
#####################################################################

DICT_ROOT = os.environ['SNORKELHOME'] + '/tutorials/disease_tagging/data/dicts/'


def load_bioportal_csv_dictionary(filename):
    '''BioPortal Ontologies:  http://bioportal.bioontology.org/'''
    reader = csv.reader(open(filename,"rU"),delimiter=',', quotechar='"')
    d = [line for line in reader]
    
    dictionary = {}
    for line in d[1:]:
        row = dict(zip(d[0],line))
        dictionary[row["Preferred Label"]] = 1
        dictionary.update({t:1 for t in row["Synonyms"].split("|")})
        
    return dictionary


def load_disease_dictionary():  
    dictfile = DICT_ROOT + "disease-names.v2.txt"
    diseases = {line.strip().split("\t")[0]:1 for line in codecs.open(dictfile, 'rb',"utf-8").readlines()}
    diseases = {word:1 for word in diseases if not word.isupper()}
        
    # Orphanet Rare Disease Ontology
    ordo = load_bioportal_csv_dictionary(DICT_ROOT + "ordo.csv")
    ordo = {word:1 for word in ordo if not word.isupper()}
    diseases.update(ordo)

    # Human Disease Ontology 
    doid = load_bioportal_csv_dictionary(DICT_ROOT + "DOID.csv")
    doid = {word:1 for word in doid if not word.isupper()}
    diseases.update(doid)

    # ------------------------------------------------------------
    # remove cell dysfunction terms
    dictfile = DICT_ROOT + "cell_molecular_dysfunction.txt"
    terms = dict.fromkeys([line.strip().split("\t")[0] for line in open(dictfile).readlines()])
    diseases = {word:1 for word in diseases if word not in terms} 

    # remove geographic terms
    dictfile = DICT_ROOT + "umls_geographic_areas.txt"
    terms = dict.fromkeys([line.strip().split("\t")[0] for line in open(dictfile).readlines()])
    diseases = {word:1 for word in diseases if word not in terms}

    # ------------------------------------------------------------
    # remove stopwords
    dictfile = DICT_ROOT + "stopwords.txt"
    stopwords = [line.strip().split("\t")[0] for line in open(dictfile).readlines()]
    diseases = {word:1 for word in diseases if word.lower() not in stopwords}  
        
    return diseases


def load_acronym_dictionary():     
    dictfile = DICT_ROOT + "disease-names.v2.txt"
    diseases = {line.strip().split("\t")[0]:1 for line in codecs.open(dictfile, 'rb', "utf-8").readlines()}
    diseases = {word:1 for word in diseases if word.isupper()}
        
    # Orphanet Rare Disease Ontology
    ordo = load_bioportal_csv_dictionary(DICT_ROOT + "ordo.csv")
    ordo = {word:1 for word in ordo if word.isupper()}
    diseases.update(ordo)
        
    # Human Disease Ontology 
    doid = load_bioportal_csv_dictionary(DICT_ROOT + "DOID.csv")
    doid = {word:1 for word in doid if word.isupper()}
    diseases.update(doid)
        
    # filter by char length
    diseases = {word:1 for word in diseases if len(word) > 1}
        
    return diseases


def load_molecular_dysfunction():
    dictfile = DICT_ROOT + "cell_molecular_dysfunction.txt"
    terms = dict.fromkeys([line.strip().split("\t")[0] for line in open(dictfile).readlines()])
    return terms


def load_syndromes(): 
    return {t.strip().lower():1 for t in codecs.open(DICT_ROOT + "syndromes.txt","rU",'utf-8').readlines()}


def load_proteins_enzymes_genes():
    return {t.strip().lower():1 for t in codecs.open(DICT_ROOT + "all.proteins_enzymes.txt","rU",'utf-8').readlines()}


def load_chemdner_dictionary():
    dict_fnames = [DICT_ROOT + "mention_chemical.txt",
                   DICT_ROOT + "chebi.txt",
                  DICT_ROOT + "addition.txt",
                 DICT_ROOT + "train.chemdner.vocab.txt"]
    chemicals = []
    for fname in dict_fnames:
        chemicals += [line.strip().split("\t")[0] for line in codecs.open(fname,"rU","utf-8").readlines()]

    # load bzip files
    fname = DICT_ROOT + "substance-sab-all.bz2"
    chemicals += {line.strip().split("\t")[0]:1 for line in bz2.BZ2File(fname, 'rb').readlines()}.keys()

    # remove stopwords
    fname = DICT_ROOT + "stopwords.txt"
    stopwords = {line.strip().split("\t")[0]:1 for line in open(fname,"rU").readlines()}
    chemicals = {term:1 for term in chemicals if term not in stopwords}
    return chemicals


def load_specialist_abbreviations():
    '''
    Load UMLS SPECIALIST Lexicon of abbreviations. Format:
    E0000048|AA|acronym|E0006859|achievement age|
    '''
    fpath = DICT_ROOT + "SPECIALIST.bz2"
    d = [line.strip().strip("|").split("|") for line in bz2.BZ2File(fpath, 'rb').readlines()]
    abbrv2text,text2abbrv =  {},{}
    for row in d:
        uid1,abbrv,atype,uid2,text = row
        text = text.lower()
        if atype not in ["acronym","abbreviation"]:
            continue
        if abbrv not in abbrv2text:
            abbrv2text[abbrv] = {}
        if text not in text2abbrv:
            text2abbrv[text] = {}

        abbrv2text[abbrv][text] = 1
        text2abbrv[text][abbrv] = 1

    return abbrv2text,text2abbrv


def load_umls_dictionary():
    umls_dict = {}
    dict_fnames = [DICT_ROOT + "snomedct.disease_or_syndrome.txt",
    DICT_ROOT + "snomedct.sign_or_symptom.txt",
    DICT_ROOT + "snomedct.finding.txt",
    DICT_ROOT + "mesh.disease_or_syndrome.txt",
    DICT_ROOT + "mesh.sign_or_symptom.txt"]

    for fname in dict_fnames:
        sab,sty = fname.split("/")[-1].split(".")[:2]
        if sab not in umls_dict:
            umls_dict[sab] = {}
        if sty not in umls_dict[sab]:
            umls_dict[sab][sty] = {}
        umls_dict[sab][sty] = {line.strip().split("\t")[0]:1 for line in codecs.open(fname, 'rU',"utf-8").readlines()}

    return umls_dict


def load_ctd_dictionary(filename, ignore_case=True):
    '''Comparative Toxicogenomics Database'''
    d = {}
    header = ['DiseaseName', 'DiseaseID', 'AltDiseaseIDs', 'Definition', 
              'ParentIDs', 'TreeNumbers', 'ParentTreeNumbers', 'Synonyms', 
              'SlimMappings']
        
    synonyms = {}
    dnames = {}
    with codecs.open(filename,"rU","utf-8",errors="ignore") as fp:
        for i,line in enumerate(fp):
            line = line.strip()
            if line[0] == "#":
                continue
            row = line.split("\t")
            if len(row) != 9:
                continue
            row = dict(zip(header,row))
             
            synonyms.update( dict.fromkeys(row["Synonyms"].strip().split("|")))
            dnames[row["DiseaseName"].strip()] = 1
    
    return {term.lower() if ignore_case else term:1 for term in synonyms.keys()+dnames.keys() if term}

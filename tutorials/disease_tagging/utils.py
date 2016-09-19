import lxml.etree as et
from snorkel.models import CandidateSet, split_stable_id
from snorkel.candidates import TemporarySpan
from collections import defaultdict
import os
import codecs
import csv
import bz2


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

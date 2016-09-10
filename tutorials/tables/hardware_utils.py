import csv
import codecs
from collections import defaultdict

def load_hardware_doc_part_pairs(filename):
    with open(filename, 'r') as csvfile:
        gold_reader = csv.reader(csvfile)
        gold = []
        for row in gold_reader:
            (doc, part, val, attr) = row
            gold.append((doc, part))
        gold = set(gold)
        return gold


def load_extended_parts_dict(filename):
    gold_pairs = load_hardware_doc_part_pairs(filename)
    (gold_docs, gold_parts) = zip(*gold_pairs)
    # make gold_parts_suffixed for matcher
    gold_parts_extended = []
    for part in gold_parts:
        for suffix in ['', 'A','B','C','-16','-25','-40']:
            gold_parts_extended.append(''.join([part,suffix]))
            if part.endswith(suffix):
                gold_parts_extended.append(part[:-len(suffix)])
                if part[:2].isalpha() and part[2:-1].isdigit() and part[-1].isalpha():
                    gold_parts_extended.append(' '.join([part[:2], part[2:-1], part[-1]]))
    # print "Loaded %s gold (doc, part) pairs." % len(gold_pairs)
    return gold_parts_extended


def load_hardware_labels(loader, candidates, filename, type, attrib):
    with codecs.open(filename, encoding="utf-8") as csvfile:
        gold_reader = csv.reader(csvfile)
        gold_dict = defaultdict(int)
        for row in gold_reader:
            (doc, part, val, attr) = row
            if attr == attrib:
                key = (doc.upper(), part.upper(), val.upper())
                gold_dict[key] += 1
                # if gold_dict[key]==2:
                #     import pdb; pdb.set_trace()
    
    for c in candidates:
        # import pdb; pdb.set_trace()
        key = ((c[0].parent.document.name).upper(), (c[0].get_span()).upper(), (''.join(c[1].get_span().split())).upper())
        if key in gold_dict:
            # import pdb; pdb.set_trace()
            # TODO: fix this hard coding
            loader.add({'part' : c[0], 'temp' : c[1]})


### TODO: DELETE ME ###
# def load_hardware_labels(loader, file_name):
#     # Get all the annotated Pubtator documents as XML trees
#     doc_xmls = get_docs_xml(ROOT + file_name)
#     for doc_id, doc_xml in doc_xmls.iteritems():
    
#         # Get the corresponding Document object
#         stable_id = "%s::document:0:0" % doc_id
#         doc       = session.query(Document).filter(Document.stable_id == stable_id).first()
#         if doc is not None:
        
#             # Use custom script + loader to add
#             for d in get_CID_unary_mentions(doc_xml, doc, 'Disease'):
#                 # d is a dictionary with {'part_temp': TemporarySpan}
#                 loader.add(d)
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


def get_gold_dict(filename, attrib, docs=None):
    with codecs.open(filename, encoding="utf-8") as csvfile:
        gold_reader = csv.reader(csvfile)
        gold_dict = defaultdict(int)
        for row in gold_reader:
            (doc, part, val, attr) = row
            if docs is None or doc.upper() in docs:
                if attr == attrib:
                    key = (doc.upper(), part.upper(), val.upper())
                    gold_dict[key] += 1
    return gold_dict

def load_hardware_labels(loader, candidates, filename, attrib, attrib_class):
    gold_dict = get_gold_dict(filename, attrib)
    
    for c in candidates:
        key = ((c[0].parent.document.name).upper(), (c[0].get_span()).upper(), (''.join(c[1].get_span().split())).upper())
        if key in gold_dict:
            loader.add({'part' : c[0], attrib_class : c[1]})

def entity_level_f1(tp, fp, tn, fn, filename, corpus, attrib):
    docs = []
    for doc in corpus:
        docs.append((doc.name).upper())
    gold_dict = get_gold_dict(filename, attrib, docs)
    
    TP = FP = TN = FN = 0
    for c in tp.union(fp):
        key = ((c[0].parent.document.name).upper(), (c[0].get_span()).upper(), (''.join(c[1].get_span().split())).upper())
        if key in gold_dict:
            TP += 1
        else:
            FP += 1
    FN = len(gold_dict) - TP

    prec = TP / float(TP + FP) if TP + FP > 0 else float('nan')
    rec  = TP / float(TP + FN) if TP + FN > 0 else float('nan')
    f1   = 2 * (prec * rec) / (prec + rec) if prec + rec > 0 else float('nan')
    print "========================================"
    print "Scoring on Entity-Level Gold Data"
    print "========================================"
    print "Corpus Precision {:.3}".format(prec)
    print "Corpus Recall    {:.3}".format(rec)
    print "Corpus F1        {:.3}".format(f1)
    print "----------------------------------------"
    print "TP: {} | FP: {} | FN: {}".format(TP, FP, FN)
    print "========================================\n"
import csv
import codecs

from snorkel.utils import ProgressBar

from snorkel.models import GoldLabel, GoldLabelKey


def get_gold_dict(filename, doc_on=True, formation_on=True, measurement_on=True, attribute=None, docs=None, integerize=False):
    attribute = [_.upper() for _ in attribute]
    with codecs.open(filename, encoding="utf-8") as csvfile:
        gold_reader = csv.reader(csvfile)
        gold_dict = set()
        for row in gold_reader:
            (doc, m1, formation, m2, measurement) = row
            if docs is None or doc.upper() in docs:
                if attribute and m2.upper() not in attribute:
                    continue
                else:
                    key = []
                    if doc_on: key.append(doc.upper())
                    if formation_on: key.append(formation.upper())
                    if measurement_on:
                        if integerize:
                            key.append(int(float(measurement)))
                        else:
                            key.append(measurement.upper())
                    gold_dict.add(tuple(key))
    return gold_dict

def load_paleo_labels(session, candidate_class, filename, attribute, annotator_name='gold'):

    ak = session.query(GoldLabelKey).filter(GoldLabelKey.name == annotator_name).first()
    if ak is None:
        ak = GoldLabelKey(name=annotator_name)
        session.add(ak)
        session.commit()

    candidates = session.query(candidate_class).all()
    gold_dict = get_gold_dict(filename, attribute=attribute)
    cand_total = len(candidates)
    print 'Loading', cand_total, 'candidate labels'
    pb = ProgressBar(cand_total)
    labels=[]
    for i, c in enumerate(candidates):
        pb.bar(i)
        doc = (c[0].sentence.document.name).upper()
        formation = (c[0].get_span()).upper()
        measurement = (''.join(c[1].get_span().split())).upper()
        context_stable_ids = '~~'.join([i.stable_id for i in c.get_contexts()])
        label = session.query(GoldLabel).filter(GoldLabel.key == ak).filter(GoldLabel.candidate == c).first()
        if label is None:
            if (doc, formation, measurement) in gold_dict:
                label = GoldLabel(candidate=c, key=ak, value=1)
            else:
                label = GoldLabel(candidate=c, key=ak, value=-1)
            session.add(label)
            labels.append(label)
    session.commit()
    pb.close()

    session.commit()
    print "AnnotatorLabels created: %s" % (len(labels),)

def entity_level_f1(candidates, gold_file, corpus, attrib):
    docs = [(doc.name).upper() for doc in corpus] if corpus else None
    gold_dict = get_gold_dict(gold_file, docs=docs, doc_on=True, formation_on=True, measurement_on=True, attribute=attrib)
    TP = FP = TN = FN = 0
    pos = set([((c[0].sentence.document.name).upper(),
                (c[0].get_span()).upper(),
                (''.join(c[1].get_span().split())).upper()) for c in candidates])
    TP_set = pos.intersection(gold_dict)
    TP = len(TP_set)
    FP_set = pos.difference(gold_dict)
    FP = len(FP_set)
    FN_set = gold_dict.difference(pos)
    FN = len(FN_set)

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
    return (TP_set, FP_set, FN_set)

def entity_to_candidates(entity, candidate_subset):
    matches = []
    for c in candidate_subset:
        c_entity = tuple([c[0].sentence.document.name.upper()] + [c[i].get_span().upper() for i in range(len(c))])
        c_entity = tuple([x.encode('utf8') for x in c_entity])
        if c_entity == entity:
            matches.append(c)
    return matches
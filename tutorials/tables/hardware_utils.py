import csv

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
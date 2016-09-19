import csv
import codecs
from collections import defaultdict
from snorkel.candidates import OmniNgrams
from snorkel.models import TemporaryImplicitSpan
from snorkel.utils import ProgressBar
from difflib import SequenceMatcher 
import re

class OmniNgramsHardware(OmniNgrams):
    def __init(self, n_max=5, split_tokens=[]):
        OmniNgrams.__init__(self, n_max=n_max, split_tokens=split_tokens)
    
    def apply(self, context):
        for ts in OmniNgrams.apply(self, context):
            part_nos = [part_no for part_no in expand_part_range(ts.get_span())]    
            if len(part_nos) == 1:
                yield ts
            else:
                for i, part_no in enumerate(part_nos):
                    yield TemporaryImplicitSpan(
                        parent         = ts.parent,
                        char_start     = 0,
                        char_end       = len(part_no) - 1,
                        expander_key   = u'part_range',
                        position       = i,
                        text           = part_no,
                        words          = [part_no],
                        char_offsets   = [0],
                        lemmas         = [part_no],
                        pos_tags       = [ts.parent.pos_tags[0]],
                        ner_tags       = [ts.parent.ner_tags[0]],
                        dep_parents    = [ts.parent.dep_parents[0]],
                        dep_labels     = [ts.parent.dep_labels[0]],
                        meta           = None
                    )
                yield ts


def load_hardware_doc_part_pairs(filename):
    with open(filename, 'r') as csvfile:
        gold_reader = csv.reader(csvfile)
        gold = []
        for row in gold_reader:
            (doc, part, val, attr) = row
            gold.append((doc.upper(), part.upper()))
        gold = set(gold)
        return gold


def load_extended_parts_dict(filename):
    gold_pairs = load_hardware_doc_part_pairs(filename)
    (gold_docs, gold_parts) = zip(*gold_pairs)
    # make gold_parts_suffixed for matcher
    gold_parts_extended = set([])
    for part in gold_parts:
        for suffix in ['', 'A','B','C','-16','-25','-40']:
            gold_parts_extended.add(''.join([part,suffix]))
            if part.endswith(suffix):
                gold_parts_extended.add(part[:-len(suffix)])
                if part[:2].isalpha() and part[2:-1].isdigit() and part[-1].isalpha():
                    gold_parts_extended.add(' '.join([part[:2], part[2:-1], part[-1]]))
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
    pb = ProgressBar(len(candidates))
    for i, c in enumerate(candidates):
        pb.bar(i)
        key = ((c[0].parent.document.name).upper(), (c[0].get_span()).upper(), (''.join(c[1].get_span().split())).upper())
        if key in gold_dict:
            loader.add({'part' : c[0], attrib_class : c[1]})
    pb.close()

def entity_level_f1(tp, fp, tn, fn, filename, corpus, attrib):
    docs = []
    for doc in corpus:
        docs.append((doc.name).upper())
    gold_dict = set(get_gold_dict(filename, attrib, docs))
    
    TP = FP = TN = FN = 0
    pos = set([((c[0].parent.document.name).upper(), 
                (c[0].get_span()).upper(), 
                (''.join(c[1].get_span().split())).upper()) for c in tp.union(fp)])
    TP = len(pos.intersection(gold_dict))
    FP = len(pos.difference(gold_dict))
    FN = len(gold_dict.difference(pos))

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


def expand_part_range(text, DEBUG=False):
    """
    Given a string, generates strings that are potentially implied by
    the original text. Two main operations are performed:
        1. Expanding ranges (X to Y; X ~ Y; X -- Y)
        2. Expanding suffixes (123X/Y/Z; 123X, Y, Z)
    If no implicit terms are found, yields just the original string.
    To get the correct output from complex strings, this function should be fed
    many Ngrams from a particular phrase.
    """
    ### Regex Patterns compile only once per function call.
    # This range pattern will find text that "looks like" a range.
    range_pattern = re.compile(ur'^(?P<start>[\w\/]+)(?:\s*(\.{3,}|\~|\-+|to|thru|through|\u2013+|\u2014+|\u2012+|\u2212+)\s*)(?P<end>[\w\/]+)$', re.IGNORECASE | re.UNICODE)
    suffix_pattern = re.compile(ur'(?P<spacer>(?:,|\/)\s*)(?P<suffix>[\w\-]+)')
    base_pattern = re.compile(ur'(?P<base>[\w\-]+)(?P<spacer>(?:,|\/)\s*)(?P<suffix>[\w\-]+)?')

    if DEBUG: print "\n[debug] Text: " + text
    expanded_parts = set()
    final_set = set()

    ### Step 1: Search and expand ranges
    m = re.search(range_pattern, text)
    if m:
        start = m.group("start")
        end = m.group("end")
        start_diff = ""
        end_diff = ""
        if DEBUG: print "[debug]   Start: %s \t End: %s" % (start, end)

        # Use difflib to find difference. We are interested in 'replace' only
        seqm = SequenceMatcher(None, start, end).get_opcodes();
        for opcode, a0, a1, b0, b1 in seqm:
            if opcode == 'equal':
                continue
            elif opcode == 'insert':
                break
            elif opcode == 'delete':
                break
            elif opcode == 'replace':
                # NOTE: Potential bug if there is more than 1 replace
                start_diff = start[a0:a1]
                end_diff = end[b0:b1]
            else:
                raise RuntimeError, "[ERROR] unexpected opcode"

        if DEBUG: print "[debug]   start_diff: %s \t end_diff: %s" % (start_diff, end_diff)

        # First, check for number range
        if atoi(start_diff) and atoi(end_diff):
            if DEBUG: print "[debug]   Enumerate %d to %d" % (atoi(start_diff), atoi(end_diff))
            # generate a list of the numbers plugged in
            for number in xrange(atoi(start_diff), atoi(end_diff) + 1):
                new_part = start.replace(start_diff,str(number))
                # Produce the strings with the enumerated ranges
                expanded_parts.add(new_part)

        # Second, check for single-letter enumeration
        if len(start_diff) == 1 and len(end_diff) == 1:
            if start_diff.isalpha() and end_diff.isalpha():
                if DEBUG: print "[debug]   Enumerate %s to %s" % (start_diff, end_diff)
                letter_range = char_range(start_diff, end_diff)
                for letter in letter_range:
                    new_part = start.replace(start_diff,letter)
                    # Produce the strings with the enumerated ranges
                    expanded_parts.add(new_part)
    else: 
        expanded_parts.add(text)
    if DEBUG: print "[debug]   Inferred Text: \n  " + str(sorted(expanded_parts))

    ### Step 2: Expand suffixes for each of the inferred phrases
    # NOTE: this only does the simple case of replacing same-length suffixes.
    # we do not handle cases like "BC546A/B/XYZ/QR"
    for part in expanded_parts:
        first_match = re.search(base_pattern, part)
        if first_match:
            base = re.search(base_pattern, part).group("base");
            final_set.add(base) # add the base (multiple times, but set handles that)
            if (first_match.group("suffix")):
                all_suffix_lengths = set()
                # This is a bit inefficient but this first pass just is here
                # to make sure that the suffixes are the same length
                for m in re.finditer(suffix_pattern, part):
                    suffix = m.group("suffix")
                    suffix_len = len(suffix)
                    all_suffix_lengths.add(suffix_len)
                if len(all_suffix_lengths) == 1:
                    for m in re.finditer(suffix_pattern, part):
                        spacer = m.group("spacer")
                        suffix = m.group("suffix")
                        suffix_len = len(suffix)
                        trimmed = base[:-suffix_len]
                        final_set.add(trimmed+suffix)
        else:
            if part and (not part.isspace()):
                final_set.add(part) # no base was found with suffixes to expand
    if DEBUG: print "[debug]   Final Set: " + str(sorted(final_set))

    # Add common part suffixes on each discovered part number
    part_suffixes = ['-16','-25','-40','A','B','C']
    # import pdb; pdb.set_trace()
    for part in final_set:
        base = part
        for suffix in part_suffixes:
            if part.endswith(suffix):
                base = part[:-len(suffix)]
                break
        if base:
            yield base
            for suffix in part_suffixes:
                yield (base + suffix).replace(' ','') # e.g., for parts in SIEMS01215-1
        else:
            yield part

    # NOTE: We make a few assumptions (e.g. suffixes must be same length), but
    # one important unstated assumption is that if there is a single suffix,
    # (e.g. BC546A/B), the single suffix will be swapped in no matter what.
    # In this example, it works. But if we had "ABCD/EFG" we would get "ABCD,AEFG"
    # Check out UtilsTests.py to see more of our assumptions capture as test
    # cases.


def atoi(num_str):
    '''
    Helper function which converts a string to an integer, or returns None.
    '''
    try:
        return int(num_str)
    except:
        pass
    return None


def char_range(a, b):
    '''
    Generates the characters from a to b inclusive.
    '''
    for c in xrange(ord(a), ord(b)+1):
        yield chr(c)


def entity_level_total_recall(total_candidates, filename, attrib):
    """Checks entity-level recall of total_candidates compared to gold.

    Turns a CandidateSet into a normal set of entity-level tuples
    (doc, part, [attrib_value])
    then compares this to the entity-level tuples found in the gold.

    Example Usage:
        from hardware_utils import entity_level_total_recall
        total_candidates = # CandidateSet of all candidates you want to consider
        filename = os.environ['SNORKELHOME'] + '/tutorials/tables/data/hardware/hardware_gold.csv'
        entity_level_total_recall(total_candidates, filename, 'stg_temp_min')
    """
    print "Preparing gold set..."
    gold_dict = get_gold_dict(filename, attrib)
    # gold_set = set(gold_dict.keys())

    gold_set = set([])
    for (doc, part, temp) in gold_dict:
        gold_set.add((doc.upper(), part.upper()))

    # Turn CandidateSet into set of tuples
    print "Preparing candidates..."
    pb = ProgressBar(len(total_candidates))
    entity_level_candidates = set()
    for i, c in enumerate(total_candidates):
        pb.bar(i)
        part = c.get_arguments()[0].get_span().upper()
        doc = c.get_arguments()[0].parent.document.name.upper()
        entity_level_candidates.add((str(doc), str(part)))
        # temp = c.get_arguments()[1].get_span()
        # doc = c.get_arguments()[1].parent.document.name
        # entity_level_candidates.add((str(doc), str(part), str(temp)))
    pb.close()

    print "========================================"
    print "Scoring on Entity-Level Total Recall"
    print "========================================"
    print "Entity-level Candidates extracted: %s " % (len(entity_level_candidates))
    print "Entity-level Gold: %s" % (len(gold_set))
    print "Intersection Candidates: %s" % (len(gold_set.intersection(entity_level_candidates)))
    print "----------------------------------------"
    print "Overlap with Gold:  %0.2f" % (len(gold_set.intersection(entity_level_candidates)) / float(len(gold_set)),)
    print "========================================\n"

    return entity_confusion_matrix(entity_level_candidates, gold_set)



def entity_confusion_matrix(pred, gold):
    if not isinstance(pred, set):
        pred = set(pred)
    if not isinstance(gold, set):
        gold = set(gold)
    TP = pred.intersection(gold)
    FP = pred.difference(gold)
    FN = gold.difference(pred)
    return (TP, FP, FN)

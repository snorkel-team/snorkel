import csv
import codecs
from collections import defaultdict
from snorkel.candidates import OmniNgrams
from snorkel.models import TemporaryImplicitSpan, CandidateSet, AnnotationKey, AnnotationKeySet, Label
from snorkel.utils import ProgressBar
from snorkel.loaders import create_or_fetch
from snorkel.throttlers import Throttler
from snorkel.lf_helpers import *
from difflib import SequenceMatcher
import re
import os

class PartThrottler(Throttler):
    """
    Removes candidates unless the part is not in a table, or the part aligned
    temperature are not aligned.
    """
    def apply(self, part_span, attr_span):
        """
        Returns True is the tuple passes, False if it should be throttled
        """
        return part_span.parent.table is None or self.aligned(part_span, attr_span)

    def aligned(self, span1, span2):
        return (span1.parent.table == span2.parent.table and
            (span1.parent.row_num == span2.parent.row_num or
             span1.parent.col_num == span2.parent.col_num))

class GainThrottler(PartThrottler):
    def apply(self, part_span, attr_span):
        """
        Returns True is the tuple passes, False if it should be throttled
        """
        return (PartThrottler.apply(self, part_span, attr_span) and
            overlap(['dc', 'gain', 'hfe', 'fe'], list(get_row_ngrams(attr_span, infer=True))))

class PartCurrentThrottler(Throttler):
    """
    Removes candidates unless the part is not in a table, or the part aligned
    temperature are not aligned.
    """
    def apply(self, part_span, current_span):
        """
        Returns True is the tuple passes, False if it should be throttled
        """
        # if both are in the same table
        if (part_span.parent.table is not None and current_span.parent.table is not None):
            if (part_span.parent.table == current_span.parent.table):
                return True

        # if part is in header, current is in table
        if (part_span.parent.table is None and current_span.parent.table is not None):
            ngrams = set(get_row_ngrams(current_span))
            # if True:
            if ('collector' in ngrams and 'current' in ngrams):
                return True

        # if neither part or current is in table
        if (part_span.parent.table is None and current_span.parent.table is None):
            ngrams = set(get_phrase_ngrams(current_span))
            num_numbers = list(get_phrase_ngrams(current_span, attrib="ner_tags")).count('number')
            if ('collector' in ngrams and 'current' in ngrams and num_numbers <= 3):
                return True

        return False

    def aligned(self, span1, span2):
        ngrams = set(get_row_ngrams(span2))
        return  (span1.parent.table == span2.parent.table and
            (span1.parent.row_num == span2.parent.row_num or span1.parent.col_num == span2.parent.col_num))


class OmniNgramsTemp(OmniNgrams):
    def __init__(self, n_max=5, split_tokens=None):
        OmniNgrams.__init__(self, n_max=n_max, split_tokens=None)

    def apply(self, context):
        for ts in OmniNgrams.apply(self, context):
            m = re.match(r'^(\+|\-|\u2010|\u2011|\u2012|\u2013|\u2014|\u2212|\%)?(\s*)(\d+)$', ts.get_span())
            if m:
                if m.group(1) is None:
                    temp = ''
                elif m.group(1) == '+':
                    if m.group(2) != '':
                        continue # If bigram '+ 150' is seen, accept the unigram '150', not both
                    temp = ''
                else:
                    # A bigram '- 150' is different from unigram '150', so we keep the implicit '-150'
                    temp = '-'
                temp += m.group(3)
                yield TemporaryImplicitSpan(
                    parent         = ts.parent,
                    char_start     = ts.char_start,
                    char_end       = ts.char_end,
                    expander_key   = u'temp_expander',
                    position       = 0,
                    text           = temp,
                    words          = [temp],
                    char_offsets   = ts.parent.char_offsets,
                    lemmas         = [temp],
                    pos_tags       = [ts.parent.pos_tags[-1]],
                    ner_tags       = [ts.parent.ner_tags[-1]],
                    dep_parents    = [ts.parent.dep_parents[-1]],
                    dep_labels     = [ts.parent.dep_labels[-1]],
                    page           = [ts.parent.page],
                    top            = [ts.parent.top[0]],
                    left           = [ts.parent.left[0]],
                    bottom         = [ts.parent.bottom[0]],
                    right          = [ts.parent.right[0]],
                    meta           = None)
            else:
                yield ts



class OmniNgramsPart(OmniNgrams):
    def __init__(self, parts_by_doc=None, n_max=5, split_tokens=None):
        # parts_by_doc is a dictionary d where d[document_name.upper()] = [partA, partB, ...]
        OmniNgrams.__init__(self, n_max=n_max, split_tokens=None)
        self.link_parts = (parts_by_doc is not None)
        self.parts_by_doc = parts_by_doc
        # using gold dictionary
        # gold_file = os.environ['SNORKELHOME'] + '/tutorials/tables/data/hardware/hardware_gold.csv'
        # gold_parts = get_gold_dict(gold_file, doc_on=True, part_on=True, val_on=False)
        # self.parts_by_doc = defaultdict(set)
        # for part in gold_parts:
        #     self.parts_by_doc[part[0]].add(part[1]) # TODO: change gold_parts to work with namedTuples
        # import pdb; pdb.set_trace()

    def apply(self, context):
        for ts in OmniNgrams.apply(self, context):
            enumerated_parts = [part_no for part_no in expand_part_range(ts.get_span())]
            if self.link_parts:
                possible_parts =  self.parts_by_doc[ts.parent.document.name.upper()]
                implicit_parts = set()
                for base in enumerated_parts:
                    for part in possible_parts:
                        if part.startswith(base):
                            implicit_parts.add(part)
            else:
                implicit_parts = set(enumerated_parts)
            for i, part_no in enumerate(implicit_parts):
                if part_no == ts.get_span():
                    yield ts
                else:
                    yield TemporaryImplicitSpan(
                        parent         = ts.parent,
                        char_start     = ts.char_start,
                        char_end       = ts.char_end,
                        expander_key   = u'part_expander',
                        position       = i,
                        text           = part_no,
                        words          = [part_no],
                        char_offsets   = ts.parent.char_offsets,
                        lemmas         = [part_no],
                        pos_tags       = [ts.parent.pos_tags[0]],
                        ner_tags       = [ts.parent.ner_tags[0]],
                        dep_parents    = [ts.parent.dep_parents[0]],
                        dep_labels     = [ts.parent.dep_labels[0]],
                        page           = [ts.parent.page],
                        top            = [ts.parent.top[0]],
                        left           = [ts.parent.left[0]],
                        bottom         = [ts.parent.bottom[0]],
                        right          = [ts.parent.right[0]],
                        meta           = None
                    )


def load_hardware_doc_part_pairs(filename):
    with open(filename, 'r') as csvfile:
        gold_reader = csv.reader(csvfile)
        gold = set()
        for row in gold_reader:
            (doc, part, val, attr) = row
            gold.add((doc.upper(), part.upper()))
        return gold


# OBSOLETE:
def load_extended_parts_dict(filename):
    gold_pairs = load_hardware_doc_part_pairs(filename)
    (gold_docs, gold_parts) = zip(*gold_pairs)
    # make gold_parts_suffixed for matcher
    gold_parts_extended = set()
    for part in gold_parts:
        for suffix in ['', 'A','B','C','-16','-25','-40']:
            gold_parts_extended.add(''.join([part,suffix]))
            if part.endswith(suffix):
                gold_parts_extended.add(part[:-len(suffix)])
                if part[:2].isalpha() and part[2:-1].isdigit() and part[-1].isalpha():
                    gold_parts_extended.add(' '.join([part[:2], part[2:-1], part[-1]]))
    return gold_parts_extended


def get_gold_parts(filename, docs=None):
    return set(map(lambda x: x[0], get_gold_dict(filename, doc_on=False, part_on=True, val_on=False, docs=docs)))


def get_gold_dict(filename, doc_on=True, part_on=True, val_on=True, attrib=None, docs=None, integerize=False):
    with codecs.open(filename, encoding="utf-8") as csvfile:
        gold_reader = csv.reader(csvfile)
        gold_dict = set()
        for row in gold_reader:
            (doc, part, val, attr) = row
            if docs is None or doc.upper() in docs:
                if attrib and attr != attrib:
                    continue
                else:
                    key = []
                    if doc_on:  key.append(doc.upper())
                    if part_on: key.append(part.upper())
                    if val_on and val:
                        if integerize:
                            key.append(int(float(val)))
                        else:
                            key.append(val.upper())
                    gold_dict.add(tuple(key))
    return gold_dict


def count_hardware_labels(candidates, filename, attrib, attrib_class):
    gold_dict = get_gold_dict(filename, attrib)
    gold_cand = defaultdict(int)
    pb = ProgressBar(len(candidates))
    for i, c in enumerate(candidates):
        pb.bar(i)
        key = ((c[0].parent.document.name).upper(), (c[0].get_span()).upper(), (''.join(c[1].get_span().split())).upper())
        if key in gold_dict:
            gold_cand[key] += 1
    pb.close()
    return gold_cand


def load_hardware_labels(session, label_set_name, annotation_key_name, candidates, filename, attrib):
    gold_dict = get_gold_dict(filename, attrib=attrib)
    candidate_set   = create_or_fetch(session, CandidateSet, label_set_name)
    annotation_key  = create_or_fetch(session, AnnotationKey, annotation_key_name)
    key_set         = create_or_fetch(session, AnnotationKeySet, annotation_key_name)
    if annotation_key not in key_set.keys:
        key_set.append(annotation_key)
    session.commit()

    cand_total = len(candidates)
    print 'Loading', cand_total, 'candidate labels'
    pb = ProgressBar(cand_total)
    for i, c in enumerate(candidates):
        pb.bar(i)
        doc = (c[0].parent.document.name).upper()
        part = (c[0].get_span()).upper()
        val = (''.join(c[1].get_span().split())).upper()
        if (doc, part, val) in gold_dict:
            candidate_set.append(c)
            session.add(Label(key=annotation_key, candidate=c, value=1))
    session.commit()
    pb.close()
    return (candidate_set, annotation_key)


def entity_level_total_recall(candidates, gold_file, attribute, relation=True, integerize=False):
    """Checks entity-level recall of candidates compared to gold.

    Turns a CandidateSet into a normal set of entity-level tuples
    (doc, part, [attribute_value])
    then compares this to the entity-level tuples found in the gold.

    Example Usage:
        from hardware_utils import entity_level_total_recall
        candidates = # CandidateSet of all candidates you want to consider
        gold_file = os.environ['SNORKELHOME'] + '/tutorials/tables/data/hardware/hardware_gold.csv'
        entity_level_total_recall(candidates, gold_file, 'stg_temp_min')
    """
    gold_set = get_gold_dict(gold_file, doc_on=True, part_on=True, val_on=relation, attrib=attribute, integerize=integerize)
    # Turn CandidateSet into set of tuples
    print "Preparing candidates..."
    pb = ProgressBar(len(candidates))
    entity_level_candidates = set()
    for i, c in enumerate(candidates):
        pb.bar(i)
        part = c.get_arguments()[0].get_span().replace(' ', '')
        doc = c.get_arguments()[0].parent.document.name
        if relation:
            if integerize:
                val = int(float(c.get_arguments()[1].get_span().replace(' ', '')))
                entity_level_candidates.add((doc.upper(), part.upper(), val))
            else:
                val = c.get_arguments()[1].get_span().replace(' ', '')
                entity_level_candidates.add((doc.upper(), part.upper(), val.upper()))
        else:
            entity_level_candidates.add((doc.upper(), part.upper()))
    pb.close()

    # import pdb; pdb.set_trace()
    print "========================================"
    print "Scoring on Entity-Level Total Recall"
    print "========================================"
    print "Entity-level Candidates extracted: %s " % (len(entity_level_candidates))
    print "Entity-level Gold: %s" % (len(gold_set))
    print "Intersection Candidates: %s" % (len(gold_set.intersection(entity_level_candidates)))
    print "----------------------------------------"
    print "Overlap with Gold:  %0.4f" % (len(gold_set.intersection(entity_level_candidates)) / float(len(gold_set)),)
    print "========================================\n"

    return entity_confusion_matrix(entity_level_candidates, gold_set)


def most_common_document(candidates):
    """Returns the document that produced the most of the passed-in candidates"""
    # Turn CandidateSet into set of tuples
    pb = ProgressBar(len(candidates))
    candidate_count = {}

    for i, c in enumerate(candidates):
        pb.bar(i)
        part = c.get_arguments()[0].get_span()
        doc = c.get_arguments()[0].parent.document.name
        candidate_count[doc] = candidate_count.get(doc, 0) + 1 # count number of occurences of keys
    pb.close()
    max_doc = max(candidate_count, key=candidate_count.get)
    return max_doc


def entity_confusion_matrix(pred, gold):
    if not isinstance(pred, set):
        pred = set(pred)
    if not isinstance(gold, set):
        gold = set(gold)
    TP = pred.intersection(gold)
    FP = pred.difference(gold)
    FN = gold.difference(pred)
    return (TP, FP, FN)


def entity_level_f1(tp, fp, tn, fn, gold_file, corpus, attrib):
    docs = []
    for doc in corpus:
        docs.append((doc.name).upper())
    gold_dict = get_gold_dict(gold_file, doc_on=True, part_on=True, val_on=True, attrib=attrib, docs=docs)

    TP = FP = TN = FN = 0
    pos = set([((c[0].parent.document.name).upper(),
                (c[0].get_span()).upper(),
                (''.join(c[1].get_span().split())).upper()) for c in tp.union(fp)])
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

        # If we cannot identify a clear number or letter range, or if there are
        # multple ranges being expressed, just ignore it.
        if len(expanded_parts) == 0:
            expanded_parts.add(text)
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

    for part in final_set:
        yield part

    # Add common part suffixes on each discovered part number
    # part_suffixes = ['-16','-25','-40','A','B','C']
    # for part in final_set:
    #     base = part
    #     for suffix in part_suffixes:
    #         if part.endswith(suffix):
    #             base = part[:-len(suffix)].replace(' ', '') # e.g., for parts in SIEMS01215-1
    #             break
    #     if base:
    #         yield base
    #         for suffix in part_suffixes:
    #             yield base + suffix
    #     else:
    #         yield part

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


def candidates_to_entities(candidates):
    entities = set()
    pb = ProgressBar(len(candidates))
    for i, c in enumerate(candidates):
        pb.bar(i)
        part = c.get_arguments()[0]
        attr = c.get_arguments()[1]
        doc  = part.parent.document.name
        entities.add((doc.upper(), part.get_span().upper(), attr.get_span().upper()))
    pb.close()
    return entities


def entity_to_candidates(entity, candidate_subset):
    matches = []
    for c in candidate_subset:
        # NOTE: should some 'upper' be going on here somewhere?
        part = c.get_arguments()[0]
        attr = c.get_arguments()[1]
        if (part.parent.document.name, part.get_span(), attr.get_span()) == entity:
            matches.append(c)
    return matches


def count_labels(entities, gold):
    T = 0
    F = 0
    for e in entities:
        if e in gold:
            T += 1
        else:
            F += 1
    return (T, F)

def part_error_analysis(c):
    print "Doc: %s" % c.part.parent.document
    print "------------"
    part = c.get_arguments()[0]
    print "Part:"
    print part
    table_info(part)
    print "------------"
    attr = c.get_arguments()[1]
    print "Attr:"
    print attr
    table_info(attr)
    print "------------"

def table_info(span):
    print "Table: %s" % span.parent.table
    if span.parent.cell:
        print "Row: %s" % span.parent.row_num
        print "Col: %s" % span.parent.col_num
    print "Phrase: %s" % span.parent

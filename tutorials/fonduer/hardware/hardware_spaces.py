from snorkel.contrib.fonduer.fonduer.candidates import OmniNgrams
from snorkel.contrib.fonduer.fonduer.models import TemporaryImplicitSpan


import re
from difflib import SequenceMatcher

def expand_part_range(text, DEBUG=False):
    """
    Given a string, generates strings that are potentially implied by
    the original text. Two main operations are performed:
        1. Expanding ranges (X to Y; X ~ Y; X -- Y)
        2. Expanding suffixes (123X/Y/Z; 123X, Y, Z)
    Also yields the original input string.
    To get the correct output from complex strings, this function should be fed
    many Ngrams from a particular phrase.
    """
    ### Regex Patterns compile only once per function call.
    # This range pattern will find text that "looks like" a range.
    range_pattern = re.compile(ur'^(?P<start>[\w\/]+)(?:\s*(\.{3,}|\~|\-+|to|thru|through|\u2011+|\u2012+|\u2013+|\u2014+|\u2012+|\u2212+)\s*)(?P<end>[\w\/]+)$', re.IGNORECASE | re.UNICODE)
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
        # multiple ranges being expressed, just ignore it.
        if len(expanded_parts) == 0:
            expanded_parts.add(text)
    else:
        expanded_parts.add(text)
        # Special case is when there is a single slack (e.g. BC337-16/BC338-16)
        # and we want to output both halves of the slash, assuming that both
        # halves are the same length
        if text.count('/') == 1:
            split = text.split('/')
            if len(split[0]) == len(split[1]):
                expanded_parts.add(split[0])
                expanded_parts.add(split[1])


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
                # first_suffix = first_match.group("suffix")
                # if part.startswith('BC547'):
                #     import pdb; pdb.set_trace()
                for m in re.finditer(suffix_pattern, part):
                    suffix = m.group("suffix")
                    suffix_len = len(suffix)
                    all_suffix_lengths.add(suffix_len)
                if len(all_suffix_lengths) == 1:
                    for m in re.finditer(suffix_pattern, part):
                        spacer = m.group("spacer")
                        suffix = m.group("suffix")
                        suffix_len = len(suffix)
                        old_suffix = base[-suffix_len:]
                        if ((suffix.isalpha() and old_suffix.isalpha()) or
                            (suffix.isdigit() and old_suffix.isdigit())):
                            trimmed = base[:-suffix_len]
                            final_set.add(trimmed+suffix)
        else:
            if part and (not part.isspace()):
                final_set.add(part) # no base was found with suffixes to expand
    if DEBUG: print "[debug]   Final Set: " + str(sorted(final_set))

    # Also return the original input string
    final_set.add(text)

    for part in final_set:
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


class OmniNgramsPart(OmniNgrams):
    def __init__(self, parts_by_doc=None, n_max=3, expand=True, split_tokens=None):
        """:param parts_by_doc: a dictionary d where d[document_name.upper()] = [partA, partB, ...]"""
        OmniNgrams.__init__(self, n_max=n_max, split_tokens=None)
        self.parts_by_doc = parts_by_doc
        self.expander = expand_part_range if expand else (lambda x: [x])

    def apply(self, session, context):
        for ts in OmniNgrams.apply(self, session, context):
            enumerated_parts = [part.upper() for part in expand_part_range(ts.get_span())]
            parts = set(enumerated_parts)
            if self.parts_by_doc:
                possible_parts = self.parts_by_doc[ts.parent.document.name.upper()]
                for base_part in enumerated_parts:
                    for part in possible_parts:
                        if part.startswith(base_part) and len(base_part) >= 4:
                            parts.add(part)
            for i, part in enumerate(parts):
                if ' ' in part:
                    continue # it won't pass the part_matcher
                # TODO: Is this try/except necessary?
                try:
                    part.decode('ascii')
                except:
                    continue
                if part == ts.get_span():
                    yield ts
                else:
                    yield TemporaryImplicitSpan(
                        sentence       = ts.sentence,
                        char_start     = ts.char_start,
                        char_end       = ts.char_end,
                        expander_key   = u'part_expander',
                        position       = i,
                        text           = part,
                        words          = [part],
                        lemmas         = [part],
                        pos_tags       = [ts.get_attrib_tokens('pos_tags')[0]],
                        ner_tags       = [ts.get_attrib_tokens('ner_tags')[0]],
                        dep_parents    = [ts.get_attrib_tokens('dep_parents')[0]],
                        dep_labels     = [ts.get_attrib_tokens('dep_labels')[0]],
                        page           = [min(ts.get_attrib_tokens('page'))] if ts.sentence.is_visual() else [None],
                        top            = [min(ts.get_attrib_tokens('top'))] if ts.sentence.is_visual() else [None],
                        left           = [max(ts.get_attrib_tokens('left'))] if ts.sentence.is_visual() else [None],
                        bottom         = [min(ts.get_attrib_tokens('bottom'))] if ts.sentence.is_visual() else [None],
                        right          = [max(ts.get_attrib_tokens('right'))] if ts.sentence.is_visual() else [None],
                        meta           = None
                    )


class OmniNgramsTemp(OmniNgrams):
    # def __init__(self, n_max=2, split_tokens=None):
    #     OmniNgrams.__init__(self, n_max=n_max, split_tokens=None)

    def apply(self, session, context):
        for ts in OmniNgrams.apply(self, session, context):
            m = re.match(u'^([\+\-\u2010\u2011\u2012\u2013\u2014\u2212\uf02d])?(\s*)(\d+)$', ts.get_span(), re.U)
            if m:
                if m.group(1) is None:
                    temp = ''
                elif m.group(1) == '+':
                    if m.group(2) != '':
                        continue # If bigram '+ 150' is seen, accept the unigram '150', not both
                    temp = ''
                else: # m.group(1) is a type of negative sign
                    # A bigram '- 150' is different from unigram '150', so we keep the implicit '-150'
                    temp = '-'
                temp += m.group(3)
                yield TemporaryImplicitSpan(
                    sentence         = ts.sentence,
                    char_start     = ts.char_start,
                    char_end       = ts.char_end,
                    expander_key   = u'temp_expander',
                    position       = 0,
                    text           = temp,
                    words          = [temp],
                    lemmas         = [temp],
                    pos_tags       = [ts.get_attrib_tokens('pos_tags')[-1]],
                    ner_tags       = [ts.get_attrib_tokens('ner_tags')[-1]],
                    dep_parents    = [ts.get_attrib_tokens('dep_parents')[-1]],
                    dep_labels     = [ts.get_attrib_tokens('dep_labels')[-1]],
                    page           = [ts.get_attrib_tokens('page')[-1]] if ts.sentence.is_visual() else [None],
                    top            = [ts.get_attrib_tokens('top')[-1]] if ts.sentence.is_visual() else [None],
                    left           = [ts.get_attrib_tokens('left')[-1]] if ts.sentence.is_visual() else [None],
                    bottom         = [ts.get_attrib_tokens('bottom')[-1]] if ts.sentence.is_visual() else [None],
                    right          = [ts.get_attrib_tokens('right')[-1]] if ts.sentence.is_visual() else [None],
                    meta           = None)
            else:
                yield ts


class OmniNgramsVolt(OmniNgrams):
    # def __init__(self, n_max=1, split_tokens=None):
    #     OmniNgrams.__init__(self, n_max=n_max, split_tokens=None)

    def apply(self, session, context):
        for ts in OmniNgrams.apply(self, session, context):
            if ts.get_span().endswith('.0'):
                value = ts.get_span()[:-2]
                yield TemporaryImplicitSpan(
                    sentence         = ts.sentence,
                    char_start     = ts.char_start,
                    char_end       = ts.char_end,
                    expander_key   = u'volt_expander',
                    position       = 0,
                    text           = value,
                    words          = [value],
                    lemmas         = [value],
                    pos_tags       = [ts.get_attrib_tokens('pos_tags')[-1]],
                    ner_tags       = [ts.get_attrib_tokens('ner_tags')[-1]],
                    dep_parents    = [ts.get_attrib_tokens('dep_parents')[-1]],
                    dep_labels     = [ts.get_attrib_tokens('dep_labels')[-1]],
                    page           = [ts.get_attrib_tokens('page')[-1]] if ts.sentence.is_visual() else [None],
                    top            = [ts.get_attrib_tokens('top')[-1]] if ts.sentence.is_visual() else [None],
                    left           = [ts.get_attrib_tokens('left')[-1]] if ts.sentence.is_visual() else [None],
                    bottom         = [ts.get_attrib_tokens('bottom')[-1]] if ts.sentence.is_visual() else [None],
                    right          = [ts.get_attrib_tokens('right')[-1]] if ts.sentence.is_visual() else [None],
                    meta           = None)
            else:
                yield ts

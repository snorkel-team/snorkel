from hardware_utils import OmniNgramsTemp

spaces = {}

matchers['part'] = OmniNgramsPart(parts_by_doc=None, n_max=3)
matchers['stg_temp_max'] = OmniNgramsTemp(n_max=2)
matchers['eb_v_max'] = OmniNgramsVolt(n_max=1)

def get_space(attr):
    return spaces[attr]


class OmniNgramsVolt(OmniNgrams):
    def __init__(self, n_max=1, split_tokens=None):
        OmniNgrams.__init__(self, n_max=n_max, split_tokens=None)    

    def apply(self, context):
        for ts in OmniNgrams.apply(self, context):
            if ts.get_span().endswith('.0'):
                value = ts.get_span()[:-2]
                yield TemporaryImplicitSpan(
                    parent         = ts.parent,
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
                    page           = [ts.get_attrib_tokens('page')[-1]] if ts.parent.is_visual() else [None],
                    top            = [ts.get_attrib_tokens('top')[-1]] if ts.parent.is_visual() else [None],
                    left           = [ts.get_attrib_tokens('left')[-1]] if ts.parent.is_visual() else [None],
                    bottom         = [ts.get_attrib_tokens('bottom')[-1]] if ts.parent.is_visual() else [None],
                    right          = [ts.get_attrib_tokens('right')[-1]] if ts.parent.is_visual() else [None],
                    meta           = None)
            else:
                yield ts


class OmniNgramsTemp(OmniNgrams):
    def __init__(self, n_max=2, split_tokens=None):
        OmniNgrams.__init__(self, n_max=n_max, split_tokens=None)

    def apply(self, context):
        for ts in OmniNgrams.apply(self, context):
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
                    parent         = ts.parent,
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
                    page           = [ts.get_attrib_tokens('page')[-1]] if ts.parent.is_visual() else [None],
                    top            = [ts.get_attrib_tokens('top')[-1]] if ts.parent.is_visual() else [None],
                    left           = [ts.get_attrib_tokens('left')[-1]] if ts.parent.is_visual() else [None],
                    bottom         = [ts.get_attrib_tokens('bottom')[-1]] if ts.parent.is_visual() else [None],
                    right          = [ts.get_attrib_tokens('right')[-1]] if ts.parent.is_visual() else [None],
                    meta           = None)
            else:
                yield ts



class OmniNgramsPart(OmniNgrams):
    def __init__(self, parts_by_doc=None, n_max=5, split_tokens=None):
        """:param parts_by_doc: a dictionary d where d[document_name.upper()] = [partA, partB, ...]"""
        OmniNgrams.__init__(self, n_max=n_max, split_tokens=None)
        self.parts_by_doc = parts_by_doc

    def apply(self, context):
        for ts in OmniNgrams.apply(self, context):
            enumerated_parts = [part.upper() for part in expand_part_range(ts.get_span())]
            parts = set(enumerated_parts)
            if self.parts_by_doc:
                possible_parts =  self.parts_by_doc[ts.parent.document.name.upper()]
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
                        parent         = ts.parent,
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
                        page           = [min(ts.get_attrib_tokens('page'))] if ts.parent.is_visual() else [None],
                        top            = [min(ts.get_attrib_tokens('top'))] if ts.parent.is_visual() else [None],
                        left           = [max(ts.get_attrib_tokens('left'))] if ts.parent.is_visual() else [None],
                        bottom         = [min(ts.get_attrib_tokens('bottom'))] if ts.parent.is_visual() else [None],
                        right          = [max(ts.get_attrib_tokens('right'))] if ts.parent.is_visual() else [None],
                        meta           = None
                    )


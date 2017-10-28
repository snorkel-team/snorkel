from ..grammar import GrammarMixin, Rule, sems0, sems1, sems_in_order, sems_reversed, flip_dir
from ..core import PrimitiveTemplate
from text_helpers import helpers
from text_annotators import annotators

lexical_rules = (
    [Rule('$Token', w, 'token') for w in ['token']] + 
    [Rule('$Word', w, 'words') for w in ['word', 'words', 'term', 'terms', 'phrase', 'phrases']] + 
    [Rule('$Char', w, 'chars') for w in ['character', 'characters', 'letter', 'letters']] + 
    [Rule('$Upper', w, '.upper') for w in ['upper', 'uppercase', 'upper case', 'all caps', 'all capitalized']] +
    [Rule('$Lower', w, '.lower') for w in ['lower', 'lowercase', 'lower case']] +
    [Rule('$Capital', w, '.capital') for w in ['capital', 'capitals', 'capitalized']] +
    [Rule('$StartsWith', w, '.startswith') for w in ['starts with', 'start with', 'starting with']] +
    [Rule('$EndsWith', w, '.endswith') for w in ['ends with', 'end with', 'ending with']] +
    [Rule('$Left', w, '.left') for w in ['?to ?the left ?of', 'in front of', 'before', 'precedes', 'preceding', 'followed by']] +
    [Rule('$Right', w, '.right') for w in ['?to ?the right ?of', 'behind', 'after', 'preceded by', 'follows', 'following']] +
    [Rule('$Within', w, '.within') for w in ['within']] +
    [Rule('$Sentence', w, '.sentence') for w in ['sentence', 'text', 'it']] +
    [Rule('$Between', w, '.between') for w in ['between', 'inbetween', 'sandwiched', 'enclosed']] +
    # TODO: Add more POS options
    [Rule('$NounPOS', w, ('NN')) for w in ['noun', 'nouns']] +
    # TODO: Add other Spacy NER options
    [Rule('$PersonNER', w, ('PERSON')) for w in ['person', 'people']] +
    [Rule('$LocationNER', w, ('LOC')) for w in ['location', 'locations', 'place', 'places']] +
    [Rule('$DateNER', w, ('DATE')) for w in ['date', 'dates']] +
    [Rule('$NumberNER', w, ('ORDINAL|CARDINAL')) for w in ['number', 'numbers']] +
    [Rule('$OrganizationNER', w, ('ORG')) for w in ['organization', 'organizations', 'company', 'companies', 'agency', 'agencies', 'institution', 'institutions']] +
    [Rule('$NorpNER', w, ('NORP')) for w in ['political', 'politician', 'religious']] +

    # FIXME: Temporary hardcode; replace with "domain_rules" passed to grammar
    [Rule('$Arg', w, '.arg') for w in ['person', 'name']] +
    [Rule('$ArgXListAnd', w, ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))) for w in ['people', 'persons', 'names']]
    # FIXME
)

unary_rules = [
    Rule('$Direction', '$Left', sems0),
    Rule('$Direction', '$Right', sems0),

    Rule('$POS', '$NounPOS', sems0),
    Rule('$POS', '$NumberPOS', sems0),
    Rule('$NER', '$DateNER', sems0),
    Rule('$NER', '$PersonNER', sems0),
    Rule('$NER', '$LocationNER', sems0),
    Rule('$NER', '$OrganizationNER', sems0),   

    Rule('$Unit', '$Word', sems0),
    Rule('$Unit', '$Char', sems0),

    # ArgX may be treated as an object or a string (referring to its textual contents)
    Rule('$String', '$ArgX', lambda sems: ('.arg_to_string', sems[0])),
    Rule('$ArgToString', '$CID', lambda sems: (sems[0],)),

    Rule('$StringListOr', '$UserList', sems0),

    Rule('$UnaryStringToBool', '$Lower', sems0),
    Rule('$UnaryStringToBool', '$Upper', sems0),
    Rule('$UnaryStringToBool', '$Capital', sems0),

    Rule('$StringBinToBool', '$Equals', sems0),
    Rule('$StringBinToBool', '$StartsWith', sems0),
    Rule('$StringBinToBool', '$EndsWith', sems0),

    # These represent string comparisons (like the letter 'a' in 'cat'), 
    # not set comparisons (like 'cat' in ['dog', 'cat', 'bird'])
    Rule('$StringBinToBool', '$In', sems0),
    Rule('$StringBinToBool', '$Contains', sems0),
]
    

compositional_rules = [
    # Text Baseline
    # NEW
    Rule('$ROOT', '$Start $Label $Bool $Because $String $Stop', 
        lambda (start_, lab_, bool_, _, str_, stop_): 
        ('.root', (lab_, bool_, ('.call', ('.in', ('.extract_text', ('.sentence',))), str_)))),
    Rule('$ROOT', '$Start $Label $Bool $Because $StringList $Stop', 
        lambda (start_, lab_, bool_, _, strlist_, stop_): 
        ('.root', (lab_, bool_, ('.all', ('.map', ('.in', ('.extract_text', ('.sentence',))), strlist_))))),
    # NEW

    # Direction
        # "is left of Y"
    Rule('$StringToBool', '$Direction $ArgX', lambda (dir_, arg_): ('.in', ('.extract_text', (dir_, arg_)))),

        # "is two words left of Y"
    Rule('$StringToBool', '$Int ?$Unit $Direction $ArgX', 
        lambda (int_, unit_, dir_, arg_): ('.in', ('.extract_text', 
            (dir_, arg_, ('.string', '.eq'), int_, ('.string', (unit_ if unit_ else 'words')))))),
        # "X is immediately before"    
    Rule('$StringToBool', '$ArgX $Int ?$Unit $Direction', 
        lambda (arg_, int_, unit_, dir_): ('.in', ('.extract_text', 
            (flip_dir(dir_), arg_, ('.string', '.eq'), int_, ('.string', (unit_ if unit_ else 'words')))))),
        
        # "is at least five words to the left of"
    Rule('$StringToBool', '$Compare $Int ?$Unit $Direction $ArgX', 
        lambda (cmp_, int_, unit_, dir_, arg_): ('.in', ('.extract_text', 
            (dir_, arg_, ('.string', cmp_), int_,('.string', (unit_ if unit_ else 'words')))))), 
        # "is to the left of Y by at least five words"
    Rule('$StringToBool', '$Direction $ArgX $Compare $Int ?$Unit', 
        lambda (dir_, arg_, cmp_, int_, unit_): ('.in', ('.extract_text', 
            (dir_, arg_, ('.string', cmp_), int_,('.string', (unit_ if unit_ else 'words')))))), 
    
    # Others
        # "is within 5 words of X"
    Rule('$StringToBool', '$Within $Int ?$Unit $ArgX', 
        lambda (win_, int_, unit_, arg_): ('.in', ('.extract_text', 
            (win_, arg_, int_, ('.string', (unit_ if unit_ else 'words')))))), 
        # "between X and Y"
    Rule('$StringToBool', '$Between $ArgXListAnd', 
        lambda (btw_, arglist_): ('.in', ('.extract_text', (btw_, arglist_)))), 
        # "in the sentence"
    Rule('$StringToBool', '$In $Sentence', 
        lambda (in_, sent_): ('.in', ('.extract_text', (sent_,)))), 
        # "sentence contains 'foo'"
    Rule('$Bool', '$Sentence $Contains $String', 
        lambda (sent_, cont_, str_): ('.call', (cont_, str_), ('.extract_text', (sent_,)))), 
    
    # Phrases
        # standard directions: "to the left of arg 1"
    Rule('$Phrase', '$Direction $ArgX', lambda (dir_, arg_): (dir_, arg_)),
    Rule('$Phrase', '$Within $ArgX', lambda (dir_, arg_): (dir_, arg_)),
    Rule('$Phrase', '$Between $ArgXListAnd', lambda (btw_, arglist_): (btw_, arglist_)),
    Rule('$Phrase', '$Sentence', lambda (sent,): (sent,)),
        
        # inverted directions: "arg 1 is right of"
    Rule('$Phrase', '$ArgX $Direction', lambda (arg_, dir_): (flip_dir(dir_), arg_)),

        # "there are three [nouns in the sentence]"
    Rule('$TokenList', '$Word $Phrase', lambda (word_, phr_): ('.filter', phr_, 'words', r'\w+\S*')),
    Rule('$TokenList', '$Char $Phrase', lambda (char_, phr_): ('.filter', phr_, 'chars', None)),
    Rule('$TokenList', '$POS $Phrase', lambda (pos_, phr_): ('.filter', phr_, 'pos_tags', pos_)),
    Rule('$TokenList', '$NER $Phrase', lambda (ner_, phr_): ('.filter', phr_, 'ner_tags', ner_)),
    Rule('$StringList', '$TokenList', sems0),

    # Count
        # "the [number of (words left of arg 1)] is larger than five"
    Rule('$Int', '$Count $Phrase', sems_in_order),         
    Rule('$Bool', '?$Exists $NumToBool $TokenList', 
        lambda (exists_, func_, list_): ('.call', func_, ('.count', list_))),

    # Arg lists
    Rule('$String', '$ArgToString $ArgX', lambda (func_, arg_): ('.call', func_, arg_)),
    Rule('$String', '$ArgX $ArgToString', lambda (arg_, func_): ('.call', func_, arg_)),
    Rule('$StringListAnd', '$ArgToString $ArgXListAnd', lambda (func_, args_): ('.map', func_, args_)),
    Rule('$StringListAnd', '$ArgXListAnd $ArgToString', lambda (args_, func_): ('.map', func_, args_)),
    
    # Tuples
    Rule('$StringTuple', '$Tuple $StringList', sems_in_order),
    Rule('$StringTupleToBool', '$Equals $StringTuple', sems_in_order),

    ### Strings ###
        # building strings of arbitrary length
    # Rule('$StringStub', '$Quote $QueryToken', lambda sems: [sems[1]]),
    # Rule('$StringStub', '$StringStub $QueryToken', lambda sems: sems[0] + [sems[1]]),
    # Rule('$String', '$StringStub $Quote', lambda sems: ('.string', ' '.join(sems[0]))),
        # building strings of max length 5 (allows us to reduce beam width)
    Rule('$String', '$Quote $QueryToken $Quote', lambda sems: ('.string', ' '.join(sems[1:2]))),
    Rule('$String', '$Quote $QueryToken $QueryToken $Quote', lambda sems: ('.string', ' '.join(sems[1:3]))),
    Rule('$String', '$Quote $QueryToken $QueryToken $QueryToken $Quote', lambda sems: ('.string', ' '.join(sems[1:4]))),
    Rule('$String', '$Quote $QueryToken $QueryToken $QueryToken $QueryToken $Quote', lambda sems: ('.string', ' '.join(sems[1:5]))),
    Rule('$String', '$Quote $QueryToken $QueryToken $QueryToken $QueryToken $QueryToken $Quote', lambda sems: ('.string', ' '.join(sems[1:6]))),

        # defining $StringToBool functions
    Rule('$StringToBool', '$UnaryStringToBool', lambda sems: (sems[0],)),
]

# template_rules = []
template_rules = (
    PrimitiveTemplate('$String') +
    PrimitiveTemplate('$StringTuple')
)

rules = lexical_rules + unary_rules + compositional_rules + template_rules

ops = {
    # string functions
    '.upper': lambda c: lambda x: lambda cx: x(cx).isupper(),
    '.lower': lambda c: lambda x: lambda cx: x(cx).islower(),
    '.capital': lambda c: lambda x: lambda cx: x(cx)[0].isupper(),
    '.startswith': lambda x: lambda cx: lambda y: lambda cy: y(cy).startswith(x(cx)),
    '.endswith': lambda x: lambda cx: lambda y: lambda cy: y(cy).endswith(x(cx)),
    # context functions
    '.arg_to_string': lambda x: lambda c: x(c).strip() if isinstance(x(c), basestring) else x(c).get_span().strip(),
    '.left': lambda *x: lambda cx: cx['helpers']['get_left_phrase'](*[xi(cx) for xi in x]),
    '.right': lambda *x: lambda cx: cx['helpers']['get_right_phrase'](*[xi(cx) for xi in x]),
    '.within': lambda *x: lambda cx: cx['helpers']['get_within_phrase'](*[xi(cx) for xi in x]),
    '.between': lambda x: lambda c: c['helpers']['get_between_phrase'](*[xi for xi in x(c)]),
    '.sentence': lambda c: c['helpers']['get_sentence_phrase'](c['candidate'][0]),
    '.extract_text': lambda phr: lambda c: getattr(phr(c), 'text').strip(),
    '.filter': lambda phr, field, val: lambda c: c['helpers']['phrase_filter'](phr(c), field, val),
    '.cid': lambda c: lambda arg: lambda cx: arg(cx).get_attrib_tokens(a='entity_cids')[0], # take the first token's CID    
}

translate_ops = {
    '.between': lambda list_: "between({})".format(list_),
    '.right': lambda *args_: "right({})".format(','.join(str(x) for x in args_)),
    '.left': lambda *args_: "left({})".format(','.join(str(x) for x in args_)),
    '.sentence': "sentence()",

    '.extract_text': lambda phr: "text({})".format(phr),
    '.filter': lambda phr, field, val: "filter({}, {}, {})".format(phr, field, val),
}

text_grammar = GrammarMixin(
    rules=rules,
    ops=ops,
    helpers=helpers,
    annotators=annotators,
    translate_ops=translate_ops,
)
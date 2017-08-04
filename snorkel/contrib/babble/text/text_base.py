from ..grammar import GrammarMixin, Rule, sems0, sems1, sems_in_order, sems_reversed, flip_dir
from text_helpers import helpers
from text_annotators import annotators

lexical_rules = (
    [Rule('$Left', w, '.left') for w in ['left', 'before', 'precedes', 'preceding', 'followed by']] +
    [Rule('$Right', w, '.right') for w in ['right', 'after', 'preceded by', 'follows', 'following']] +
    [Rule('$Sentence', w, '.sentence') for w in ['sentence', 'text', 'it']] +
    [Rule('$Between', w, '.between') for w in ['between', 'inbetween', 'sandwiched', 'enclosed']] +
    [Rule('$NounPOS', w, ('.string', 'NN')) for w in ['noun', 'nouns']] +
    [Rule('$DateNER', w, ('.string', 'DATE')) for w in ['date', 'dates']] +
    [Rule('$NumberPOS', w, ('.string', 'CD')) for w in ['number', 'numbers']] +
    [Rule('$PersonNER', w, ('.string', 'PERSON')) for w in ['person', 'people']] +
    [Rule('$LocationNER', w, ('.string', 'LOCATION')) for w in ['location', 'locations', 'place', 'places']] +
    [Rule('$OrganizationNER', w, ('.string', 'ORGANIZATION')) for w in ['organization', 'organizations']]
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

    Rule('$StringList', '$UserList', sems0),
    Rule('$StringList', 'StringListStub', sems0),
    Rule('$StringList', 'StringListOr', sems0),
    Rule('$StringList', 'StringListAnd', sems0),

    Rule('$UnaryStringToBool', '$Lower', sems0),
    Rule('$UnaryStringToBool', '$Upper', sems0),
    Rule('$UnaryStringToBool', '$Capital', sems0),
    Rule('$BinaryStringToBool', '$StartsWith', sems0),
    Rule('$BinaryStringToBool', '$EndsWith', sems0),

    Rule('$BinaryStringToBool', '$In', sems0),
    Rule('$BinaryStringToBool', '$Contains', sems0),
    Rule('$BinaryStringToBool', '$Equals', sems0),
]
    
compositional_rules = [
            # make lists
    Rule('$PhraseList', '$Direction $ArgX', lambda (dir_, arg_): (dir_, arg_)),
    Rule('$PhraseList', '$Between $ArgXAnd', lambda (btw_, arglist_): (btw_, arglist_)),
    Rule('$PhraseList', '$Sentence', lambda (sent,): (sent,)),

        # inverted directions
    Rule('$PhraseList', '$ArgX $Direction', lambda (arg_, dir_): (flip_dir(dir_), arg_)),

    ### Direction ###
        # "is left of Y"
    Rule('$StringToBool', '$Direction $ArgX',
        lambda (dir_, arg_): ('.in', ('.extract_text', (dir_, arg_)))),

        # "is two words left of Y"
    Rule('$StringToBool', '$Int ?$Unit $Direction $ArgX', 
        lambda (int_, unit_, dir_, arg_): ('.in', ('.extract_text', 
            (dir_, arg_, ('.string', '.eq'), int_, ('.string', (unit_ if unit_ else 'words')))))),
        # "X is immediately before"    
    Rule('$StringToBool', '$ArgX $Int ?$Unit $Direction', 
        lambda (arg_, int_, unit_, dir_): ('.in', ('.extract_text', 
            (flip_dir(dir_), arg_, ('.string', '.eq'), int_, ('.string', (unit_ if unit_ else 'words')))))),
        
        # "is at least 40 words to the left of"
    Rule('$StringToBool', '$Compare $Int ?$Unit $Direction $ArgX', 
        lambda (cmp_, int_, unit_, dir_, arg_): ('.in', ('.extract_text', 
            (dir_, arg_, ('.string', cmp_), int_,('.string', (unit_ if unit_ else 'words')))))), 
        # "is to the left of Y by at least 40 words"
    Rule('$StringToBool', '$Direction $ArgX $Compare $Int ?$Unit', 
        lambda (dir_, arg_, cmp_, int_, unit_): ('.in', ('.extract_text', 
            (dir_, arg_, ('.string', cmp_), int_,('.string', (unit_ if unit_ else 'words')))))), 
        
        # "between X and Y"
    Rule('$StringToBool', '$Between $ArgXAnd', 
        lambda (btw_, arglist_): 
            ('.in', ('.extract_text', (btw_, arglist_)))), 
        
    # Count
            # "the number of (words left of arg 1) is 5"
    Rule('$Int', '$Count $TokenList', sems_in_order),
            # "at least one word is to the left..."
    Rule('$Bool', '$NumToBool $Word $Exists $TokenList', lambda (func_, word_, exist_, list_): 
        ('.call', func_, ('.count', list_))),
            # "at least one noun is to the left..."
    Rule('$Bool', '$NumToBool $POS $Exists $TokenList', lambda sems: 
        ('.call', sems[0], ('.count', ('.filter_by_attr', sems[3], ('.string', 'pos_tags'), sems[1])))),
            # "at least one person is to the left..."
    Rule('$Bool', '$NumToBool $NER $Exists $TokenList', lambda sems: 
        ('.call', sems[0], ('.count', ('.filter_by_attr', sems[3], ('.string', 'ner_tags'), sems[1])))), 
            # "there are not three people to the left..."
    Rule('$Bool', '$Exists $Not $Int $TokenList', lambda sems: ('.call', ('.neq', sems[2]), ('.count', sems[3]))), 
            # "there are three nouns to the left..."
    Rule('$Bool', '$Exists $Int $TokenList', lambda sems: ('.call', ('.eq', sems[1]), ('.count', sems[2]))), 
            # "there are at least two nouns to the left..."
    Rule('$Bool', '$Exists $NumToBool $TokenList', lambda sems: ('.call', sems[1], ('.count', sems[2]))),

    # Arg lists
    Rule('$String', '$ArgToString $ArgX', lambda (func_, arg_): ('.call', func_, arg_)),
    Rule('$String', '$ArgX $ArgToString', lambda (arg_, func_): ('.call', func_, arg_)),
    Rule('$StringListAnd', '$ArgToString $ArgXAnd', lambda (func_, args_): ('.map', func_, args_)),
    Rule('$StringListAnd', '$ArgXAnd $ArgToString', lambda (args_, func_): ('.map', func_, args_)),
    
    # Tuples
    Rule('$StringTuple', '$Tuple $StringListAnd', sems_in_order),
    Rule('$StringTupleToBool', '$In $List', sems_in_order),
    Rule('$StringTupleToBool', '$Equals $StringTuple', sems_in_order),
    Rule('$Bool', '$StringTuple $StringTupleToBool', lambda (tup_, func_): ('.call', func_, tup_)),

    Rule('$StringTupleListStub', '$StringTuple ?$Separator $StringTuple', lambda sems: ('.list', sems[0], sems[2])),
    Rule('$StringTupleListStub', '$StringTupleListStub ?$Separator $StringTuple', lambda sems: tuple((list(sems[0]) + [sems[2]]))),
    
    Rule('$StringTupleListOr', '$StringTuple ?$Separator $Or $StringTuple', lambda sems: ('.list', sems[0], sems[3])),
    Rule('$StringTupleListOr', '$StringTupleListStub ?$Separator $Or $StringTuple', lambda sems: tuple(list(sems[0]) + [sems[3]])),

    Rule('$StringTupleListAnd', '$StringTuple ?$Separator $And $StringTuple', lambda sems: ('.list', sems[0], sems[3])),
    Rule('$StringTupleListAnd', '$StringTupleListStub ?$Separator $And $StringTuple', lambda sems: tuple(list(sems[0]) + [sems[3]])),

    Rule('$Bool', '$StringTuple $Not $StringTupleToBool', lambda (tup_, not_, func_): (not_, ('.call', func_, tup_))),
    Rule('$Bool', '$StringTupleListOr $StringTupleToBool', lambda (tuplist_, func_): ('.any', ('.map', func_, tuplist_))),
    Rule('$Bool', '$StringTupleListAnd $StringTupleToBool', lambda (tuplist_, func_): ('.all', ('.map', func_, tuplist_))),
    
    # NER/POS
    Rule('$PhraseList', '$POS $PhraseList', lambda sems: ('.filter_by_attr', sems[1], ('.string', 'pos_tags'), sems[0])),
    Rule('$PhraseList', '$NER $PhraseList', lambda sems: ('.filter_by_attr', sems[1], ('.string', 'ner_tags'), sems[0])),
    Rule('$TokenList', '$PhraseList', lambda sems: ('.filter_to_tokens', sems[0])),
    Rule('$StringList', '$PhraseList', lambda sems: ('.extract_text', sems[0])),

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

        # building string lists
    Rule('$StringListStub', '$String ?$Separator $String', lambda sems: ('.list', sems[0], sems[2])),
    Rule('$StringListStub', '$StringListStub ?$Separator $String', lambda sems: tuple((list(sems[0]) + [sems[2]]))),

    Rule('$StringListOr', '$String ?$Separator $Or $String', lambda sems: ('.list', sems[0], sems[3])),
    Rule('$StringListOr', '$StringListStub ?$Separator $Or $String', lambda sems: tuple(list(sems[0]) + [sems[3]])),
    Rule('$StringListOr', '$OpenParen $StringListStub $CloseParen', sems1),

    Rule('$StringListAnd', '$String ?$Separator $And $String', lambda sems: ('.list', sems[0], sems[3])),
    Rule('$StringListAnd', '$StringListStub ?$Separator $And $String', lambda sems: tuple(list(sems[0]) + [sems[3]])),
    Rule('$StringListAnd', '$OpenParen $StringListStub $CloseParen', sems1),

        # applying $StringToBool functions
    Rule('$Bool', '$String $StringToBool', lambda (str_, func_): ('.call', func_, str_)),
    Rule('$Bool', '$String $Not $StringToBool', lambda (str_, not_, func_): (not_, ('.call', func_, str_))),
    Rule('$Bool', '$StringListOr $StringToBool', lambda (strlist_, func_): ('.any', ('.map', func_, strlist_))),
    Rule('$Bool', '$StringListAnd $StringToBool', lambda (strlist_, func_): ('.all', ('.map', func_, strlist_))),
    Rule('$BoolList', '$StringList $StringToBool', lambda (strlist_, func_): ('.map', func_, strlist_)),
    Rule('$Bool', '$Exists $StringList $StringToBool', lambda (exists_, strlist_, func_): ('.any', ('.map', func_, strlist_))),
    Rule('$Bool', '$StringList $Exists $StringToBool', lambda (strlist_, exists_, func_): ('.any', ('.map', func_, strlist_))),

        # applying inverted $StringToBool functions
    Rule('$Bool', '$StringToBool $String', lambda (func_, str_): ('.call', func_, str_)),
    Rule('$Bool', '$StringToBool $String $Not', lambda (func_, str_, not_): (not_, ('.call', func_, str_))),
    Rule('$Bool', '$StringToBool $StringListOr', lambda (func_, strlist_): ('.any', ('.map', func_, strlist_))),
    Rule('$Bool', '$StringToBool $StringListAnd', lambda (func_, strlist_): ('.all', ('.map', func_, strlist_))),
    Rule('$BoolList', '$StringToBool $StringList ', lambda (func_, strlist_): ('.map', func_, strlist_)),
    Rule('$Bool', '$StringToBool $Exists $StringList ', lambda (func_, exists_, strlist_): ('.any', ('.map', func_, strlist_))),

        # defining $StringToBool functions
    Rule('$StringToBool', '$UnaryStringToBool', lambda sems: (sems[0],)),
    Rule('$StringToBool', '$BinaryStringToBool $String', sems_in_order),
    Rule('$StringToBool', '$In $StringList', sems_in_order),
    Rule('$StringToBool', '$BinaryStringToBool $StringListAnd', lambda sems: ('.composite_and', (sems[0],), sems[1])),
    Rule('$StringToBool', '$BinaryStringToBool $StringListOr', lambda sems: ('.composite_or',  (sems[0],), sems[1])),
    Rule('$StringToBool', '$BinaryStringToBool $UserList', lambda sems: ('.composite_or',  (sems[0],), sems[1])),  
]

rules = lexical_rules + unary_rules + compositional_rules

ops = {
    # string functions
    '.upper': lambda c: lambda x: lambda cx: x(cx).isupper(),
    '.lower': lambda c: lambda x: lambda cx: x(cx).islower(),
    '.capital': lambda c: lambda x: lambda cx: x(cx)[0].isupper(),
    '.startswith': lambda x: lambda cx: lambda y: lambda cy: y(cy).startswith(x(cx)),
    '.endswith': lambda x: lambda cx: lambda y: lambda cy: y(cy).endswith(x(cx)),
    # context functions
    '.arg_to_string': lambda x: lambda c: x(c).strip() if isinstance(x(c), basestring) else x(c).get_span().strip(),
    '.left': lambda *x: lambda cx: cx['helpers']['get_left_phrases'](*[xi(cx) for xi in x]),
    '.right': lambda *x: lambda cx: cx['helpers']['get_right_phrases'](*[xi(cx) for xi in x]),
    '.between': lambda x: lambda c: c['helpers']['get_between_phrases'](*[xi for xi in x(c)]),
    '.sentence': lambda c: c['helpers']['get_sentence_phrases'](c['candidate'][0]),
    '.extract_text': lambda phrlist: lambda c: [getattr(p, 'text').strip() for p in phrlist(c)],
    '.filter_by_attr': lambda phrlist, attr, val: lambda c: [p for p in phrlist(c) if getattr(p, attr(c))[0] == val(c)],
    '.filter_to_tokens': lambda phrlist: lambda c: [p for p in phrlist(c) if len(getattr(p, 'words')) == 1], 
    '.cid': lambda c: lambda arg: lambda cx: arg(cx).get_attrib_tokens(a='entity_cids')[0], # take the first token's CID    
}

text_grammar = GrammarMixin(
    rules=rules,
    ops=ops,
    helpers=helpers,
    annotators=annotators
)
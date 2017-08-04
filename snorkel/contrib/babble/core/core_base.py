from __future__ import print_function

from ..grammar import GrammarMixin, Rule, sems0, sems1, sems_in_order, sems_reversed, flip_dir
from core_helpers import helpers
from core_annotators import annotators

# Rules ======================================================================
lexical_rules = (
    [Rule('$Start', w) for w in ['<START>']] +
    [Rule('$Stop', w) for w in ['<STOP>']] +
    [Rule('$Label', w, '.label') for w in ['label ?it']] +
    [Rule('$Arg', w, '.arg') for w in ['arg', 'argument']] +
    [Rule('$True', w, ('.bool', True)) for w in ['true', 'correct']] +
    [Rule('$False', w, ('.bool', False)) for w in ['false', 'incorrect', 'wrong']] +
    [Rule('$And', w, '.and') for w in ['and']] +
    [Rule('$Or', w, '.or') for w in ['or', 'nor']] +
    [Rule('$Not', w, '.not') for w in ['not', "n't"]] +
    [Rule('$All', w, '.all') for w in ['all']] +
    [Rule('$Any', w, '.any') for w in ['any', 'a']] +
    [Rule('$None', w, '.none') for w in ['none', 'not any', 'neither', 'no']] +
    [Rule('$Is', w) for w in ['is', 'are', 'be', 'comes', 'appears', 'occurs']] +
    [Rule('$Exists', w) for w in ['exist', 'exists']] +
    [Rule('$Int', w, ('.int', 0)) for w in ['no']] +
    [Rule('$Int', w,  ('.int', 1)) for w in ['immediately', 'right']] +
    [Rule('$AtLeastOne', 'a', ('.geq', ('.int', 1)))] +
    # [Rule('$Int', 'a', ('.int', 1))] +
    [Rule('$Because', w) for w in ['because', 'since', 'if']] +
    [Rule('$Upper', w, '.upper') for w in ['upper', 'uppercase', 'upper case', 'all caps', 'all capitalized']] +
    [Rule('$Lower', w, '.lower') for w in ['lower', 'lowercase', 'lower case']] +
    [Rule('$Capital', w, '.capital') for w in ['capital', 'capitals', 'capitalized']] +
    [Rule('$Equals', w, '.eq') for w in ['equal', 'equals', '=', '==', 'same', 'identical', 'exactly']] + 
    [Rule('$LessThan', w, '.lt') for w in ['less than', 'smaller than', '<']] +
    [Rule('$AtMost', w, '.leq') for w in ['at most', 'no larger than', 'less than or equal', 'within', 'no more than', '<=']] +
    [Rule('$AtLeast', w, '.geq') for w in ['at least', 'no less than', 'no smaller than', 'greater than or equal', '>=']] +
    [Rule('$MoreThan', w, '.gt') for w in ['more than', 'greater than', 'larger than', '>']] + 
    [Rule('$Within', w, '.within') for w in ['within']] +
    [Rule('$In', w, '.in') for w in ['in']] +
    [Rule('$Contains', w, '.contains') for w in ['contains', 'contain', 'containing', 'include', 'includes', 'says', 'states']] +
    [Rule('$StartsWith', w, '.startswith') for w in ['starts with', 'start with', 'starting with']] +
    [Rule('$EndsWith', w, '.endswith') for w in ['ends with', 'end with', 'ending with']] +
    # [Rule('$Left', w, '.left') for w in ['left', 'before', 'precedes', 'preceding', 'followed by']] +
    # [Rule('$Right', w, '.right') for w in ['right', 'after', 'preceded by', 'follows', 'following']] +
    [Rule('$Sentence', w, '.sentence') for w in ['sentence', 'text', 'it']] +
    [Rule('$Between', w, '.between') for w in ['between', 'inbetween', 'sandwiched', 'enclosed']] +
    [Rule('$Separator', w) for w in [',', ';', '/']] +
    [Rule('$Count', w, '.count') for w in ['number', 'length', 'count']] +
    [Rule('$Word', w, 'words') for w in ['word', 'words', 'term', 'terms', 'phrase', 'phrases']] + 
    [Rule('$Char', w, 'chars') for w in ['character', 'characters', 'letter', 'letters']] + 
    [Rule('$NounPOS', w, ('.string', 'NN')) for w in ['noun', 'nouns']] +
    [Rule('$DateNER', w, ('.string', 'DATE')) for w in ['date', 'dates']] +
    [Rule('$NumberPOS', w, ('.string', 'CD')) for w in ['number', 'numbers']] +
    [Rule('$PersonNER', w, ('.string', 'PERSON')) for w in ['person', 'people']] +
    [Rule('$LocationNER', w, ('.string', 'LOCATION')) for w in ['location', 'locations', 'place', 'places']] +
    [Rule('$OrganizationNER', w, ('.string', 'ORGANIZATION')) for w in ['organization', 'organizations']] +
    [Rule('$Punctuation', w) for w in ['.', ',', ';', '!', '?']] +
    [Rule('$Tuple', w, '.tuple') for w in ['pair', 'tuple']] +

    # FIXME: Temporary hardcode
    [Rule('$ChemicalEntity', w, ('.string', 'Chemical')) for w in ['chemical', 'chemicals']] +
    [Rule('$DiseaseEntity', w, ('.string', 'Disease')) for w in ['disease', 'diseases']] +
    [Rule('$CID', w, '.cid') for w in ['cid', 'cids', 'canonical id', 'canonical ids']] +
    [Rule('$ArgXAnd', w, ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))) for w in ['them']] +
    [Rule('$Arg', w, '.arg') for w in ['person']] +
    [Rule('$ArgXAnd', w, ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))) for w in ['people', 'persons']]
    # FIXME
)

unary_rules = [
    # FIXME: Temporary hardcode
    Rule('$ArgX', '$ChemicalEntity', ('.arg', ('.int', 1))),
    Rule('$ArgX', '$DiseaseEntity', ('.arg', ('.int', 2))),
    # FIXME
    Rule('$Bool', '$BoolLit', sems0),
    Rule('$BoolLit', '$True', sems0),
    Rule('$BoolLit', '$False', sems0),
    Rule('$Num', '$Int', sems0),
    Rule('$Num', '$Float', sems0),
    Rule('$Conj', '$And', sems0),
    Rule('$Conj', '$Or', sems0),
    Rule('$Exists', '$Is'),
    Rule('$Equals', '$Is', '.eq'),
    Rule('$Compare', '$Equals', sems0),
    Rule('$Compare', '$NotEquals', sems0),
    Rule('$Compare', '$LessThan', sems0),
    Rule('$Compare', '$AtMost', sems0),
    Rule('$Compare', '$MoreThan', sems0),
    Rule('$Compare', '$AtLeast', sems0),
    Rule('$WithIO', '$Within', sems0),
    Rule('$WithIO', '$Without', sems0),
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
    Rule('$StringList', '$UserList', sems0),
    Rule('$UnaryStringToBool', '$Lower', sems0),
    Rule('$UnaryStringToBool', '$Upper', sems0),
    Rule('$UnaryStringToBool', '$Capital', sems0),
    Rule('$BinaryStringToBool', '$StartsWith', sems0),
    Rule('$BinaryStringToBool', '$EndsWith', sems0),
    Rule('$BinaryStringToBool', '$In', sems0),
    Rule('$BinaryStringToBool', '$Contains', sems0),
    Rule('$BinaryStringToBool', '$Equals', sems0),
    Rule('$NumToBool', '$AtLeastOne', sems0),
    # ArgX may be treated as an object or a string (referring to its textual contents)
    Rule('$String', '$ArgX', lambda sems: ('.arg_to_string', sems[0])),
    Rule('$ArgToString', '$CID', lambda sems: (sems[0],)),
    Rule('$StringList', 'StringListStub', sems0),
    Rule('$StringList', 'StringListOr', sems0),
    Rule('$StringList', 'StringListAnd', sems0),
    Rule('$List', '$BoolList', sems0),
    Rule('$List', '$StringList', sems0), # Also: UserList ->  StringList -> List
    Rule('$List', '$IntList', sems0),
    Rule('$List', '$TokenList', sems0),
]

compositional_rules = [
    Rule('$ROOT', '$Start $LF $Stop', lambda sems: ('.root', sems[1])),
    Rule('$LF', '$Label $Bool $Because $Bool ?$Punctuation', lambda sems: (sems[0], sems[1], sems[3])),

    ### Logicals ###
    Rule('$Bool', '$Bool $Conj $Bool', lambda sems: (sems[1], sems[0], sems[2])),
    Rule('$Bool', '$Not $Bool', sems_in_order),
    Rule('$Bool', '$All $BoolList', sems_in_order),
    Rule('$Bool', '$Any $BoolList', sems_in_order),
    Rule('$Bool', '$None $BoolList', sems_in_order),
    Rule('$Bool', '$BoolList', lambda (boollist_,): ('.any', boollist_)),

    # Parentheses
    Rule('$Bool', '$OpenParen $Bool $CloseParen', lambda (open_, bool_, close_): bool_),

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
    
    ### Integers ###
        # applying $NumToBool functions
    Rule('$Bool', '$Num $NumToBool', lambda sems: ('.call', sems[1], sems[0])),
    Rule('$BoolList', '$NumList $NumToBool', lambda sems: ('.map', sems[1], sems[0])),
    Rule('$NumToBool', '$Compare $Num', sems_in_order),

        # flipping inequalities
    Rule('$AtMost', '$Not $MoreThan', '.leq'),
    Rule('$AtLeast', '$Not $LessThan', '.geq'),
    Rule('$LessThan', '$Not $AtLeast', '.lt'),
    Rule('$MoreThan', '$Not $AtMost', '.gt'),
    Rule('$NotEquals', '$Not $Equals', '.neq'),
    Rule('$NotEquals', '$Equals $Not', '.neq'), # necessary because 'not' requires a bool, not an NumToBool
    Rule('$Without', '$Not $Within', '.without'), # necessary because 'not' requires a bool, not an NumToBool
    
        # "more than five of X words are upper"
    Rule('$Bool', '$NumToBool $BoolList', lambda (func_,boollist_): ('.call', func_, ('.sum', boollist_))),

    ### Context ###
    Rule('$ArgX', '$Arg $Int', sems_in_order),
    Rule('$ArgXOr', '$ArgX $Or $ArgX', lambda (arg1_, and_, arg2_): ('.list', arg1_, arg2_)),
    Rule('$ArgXAnd', '$ArgX $And $ArgX', lambda (arg1_, and_, arg2_): ('.list', arg1_, arg2_)),

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
    
    # NER/POS
    Rule('$PhraseList', '$POS $PhraseList', lambda sems: ('.filter_by_attr', sems[1], ('.string', 'pos_tags'), sems[0])),
    Rule('$PhraseList', '$NER $PhraseList', lambda sems: ('.filter_by_attr', sems[1], ('.string', 'ner_tags'), sems[0])),
    Rule('$TokenList', '$PhraseList', lambda sems: ('.filter_to_tokens', sems[0])),
    Rule('$StringList', '$PhraseList', lambda sems: ('.extract_text', sems[0])),

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
]

rules = lexical_rules + unary_rules + compositional_rules

ops = {
    # root
    '.root': lambda x: lambda c: x(c),
    '.label': lambda x, y: lambda c: (1 if x(c)==True else -1) if y(c)==True else 0,
    # primitives
    '.bool': lambda x: lambda c: x,
    '.string': lambda x: lambda c: x,
    '.int': lambda x: lambda c: x,
    # lists
    '.tuple': lambda x: lambda c: tuple(x(c)),
    '.list': lambda *x: lambda c: [z(c) for z in x],
    '.user_list': lambda x: lambda c: c['user_lists'][x(c)],
        # apply a function x to elements in list y
    '.map': lambda func_, list_: lambda cxy: [func_(cxy)(lambda c: yi)(cxy) for yi in list_(cxy)],
        # call a 'hungry' evaluated function on one or more arguments
    '.call': lambda *x: lambda c: x[0](c)(x[1])(c), #TODO: extend to more than one argument?
        # apply an element to a list of functions (then call 'any' or 'all' to convert to boolean)
    '.composite_and': lambda x, y: lambda cxy: lambda z: lambda cz: all([x(lambda c: yi)(cxy)(z)(cz)==True for yi in y(cxy)]),
    '.composite_or':  lambda x, y: lambda cxy: lambda z: lambda cz: any([x(lambda c: yi)(cxy)(z)(cz)==True for yi in y(cxy)]),
    # logic
        # NOTE: and/or expect individual inputs, not/all/any/none expect a single iterable of inputs
    '.and': lambda x, y: lambda c: x(c)==True and y(c)==True, 
    '.or': lambda x, y: lambda c: x(c)==True or y(c)==True,
    '.not': lambda x: lambda c: not x(c)==True,
    '.all': lambda x: lambda c: all(xi==True for xi in x(c)),
    '.any': lambda x: lambda c: any(xi==True for xi in x(c)),
    '.none': lambda x: lambda c: not any(xi==True for xi in x(c)),
    # comparisons
    '.eq': lambda x: lambda cx: lambda y: lambda cy: y(cy) == x(cx),
    '.neq': lambda x: lambda cx: lambda y: lambda cy: y(cy) != x(cx),
    '.lt': lambda x: lambda cx: lambda y: lambda cy: y(cy) < x(cx),
    '.leq': lambda x: lambda cx: lambda y: lambda cy: y(cy) <= x(cx),
    '.geq': lambda x: lambda cx: lambda y: lambda cy: y(cy) >= x(cx),
    '.gt': lambda x: lambda cx: lambda y: lambda cy: y(cy) > x(cx),
    # string functions
    '.upper': lambda c: lambda x: lambda cx: x(cx).isupper(),
    '.lower': lambda c: lambda x: lambda cx: x(cx).islower(),
    '.capital': lambda c: lambda x: lambda cx: x(cx)[0].isupper(),
    '.startswith': lambda x: lambda cx: lambda y: lambda cy: y(cy).startswith(x(cx)),
    '.endswith': lambda x: lambda cx: lambda y: lambda cy: y(cy).endswith(x(cx)),
    # lists
    '.in': lambda x: lambda cx: lambda y: lambda cy: y(cy) in x(cx),
    '.contains': lambda x: lambda cx: lambda y: lambda cy: x(cx) in y(cy),
    '.count': lambda x: lambda c: len(x(c)),
    '.sum': lambda x: lambda c: sum(x(c)),
    '.intersection': lambda x, y: lambda c: list(set(x(c)).intersection(y(c))),
    # context
    '.arg': lambda x: lambda c: c['candidate'][x(c) - 1],
        # NOTE: For ease of testing, temporarily allow tuples of strings in place of legitimate candidates
    '.arg_to_string': lambda x: lambda c: x(c).strip() if isinstance(x(c), basestring) else x(c).get_span().strip(),
    '.cid': lambda c: lambda arg: lambda cx: arg(cx).get_attrib_tokens(a='entity_cids')[0], # take the first token's CID
    # sets
    '.left': lambda *x: lambda cx: cx['helpers']['get_left_phrases'](*[xi(cx) for xi in x]),
    '.right': lambda *x: lambda cx: cx['helpers']['get_right_phrases'](*[xi(cx) for xi in x]),
    '.between': lambda x: lambda c: c['helpers']['get_between_phrases'](*[xi for xi in x(c)]),
    '.sentence': lambda c: c['helpers']['get_sentence_phrases'](c['candidate'][0]),
    '.extract_text': lambda phrlist: lambda c: [getattr(p, 'text').strip() for p in phrlist(c)],
    '.filter_by_attr': lambda phrlist, attr, val: lambda c: [p for p in phrlist(c) if getattr(p, attr(c))[0] == val(c)],
    '.filter_to_tokens': lambda phrlist: lambda c: [p for p in phrlist(c) if len(getattr(p, 'words')) == 1],
    }


def sem_to_str(sem):
    str_ops = {
        '.root': lambda LF: recurse(LF),
        '.label': lambda label, cond: "return {} if {} else 0".format(1 if recurse(label)=='True' else -1, recurse(cond)),
        '.bool': lambda bool_: bool_=='True',
        '.string': lambda str_: "'{}'".format(str_),
        '.int': lambda int_: int(int_),
        
        '.tuple': lambda list_: "tuple({})".format(recurse(list_)),
        '.list': lambda *elements: "[{}]".format(','.join(recurse(x) for x in elements)),
        '.user_list': lambda name: "${}".format(str(name)),
        '.map': lambda func_, list_: "map({}, {})".format(recurse(func_), recurse(list_)),
        '.call': lambda func_, args_: "call({}, {})".format(recurse(func_), recurse(args_)),

        '.and': lambda x, y: "({} and {})".format(recurse(x), recurse(y)),
        '.or': lambda x, y: "({} or {})".format(recurse(x), recurse(y)),
        '.not': lambda x: "not ({})".format(recurse(x)),
        '.all': lambda x: "all({})".format(recurse(x)),
        '.any': lambda x: "any({})".format(recurse(x)),
        '.none': lambda x: "not any({})".format(recurse(x)),

        # '.composite_and': lambda func_, list_: "all(map({}, {}))".format(recurse(func_), recurse(list_)),
        # '.composite_or':  lambda x, y, z: lambda cz: any([x(lambda c: yi)(cxy)(z)(cz)==True for yi in y(cxy)]),

        '.eq': lambda x: "(= {})".format(recurse(x)),
        '.geq': lambda x: "(>= {})".format(recurse(x)),

        '.arg': lambda int_: "arg{}".format(int_),
        '.arg_to_string': lambda arg_: "text({}).strip()".format(recurse(arg_)),
        '.cid': lambda arg_: "cid({})".format(recurse(arg_)),

        '.in': lambda rhs: "in {}".format(recurse(rhs)),
        '.contains': lambda rhs: "contains({})".format(recurse(rhs)),
        '.count': lambda list_: "count({})".format(recurse(list_)),
        '.sum': lambda arg_: "sum({})".format(recurse(arg_)),

        '.between': lambda list_: "between({})".format(recurse(list_)),
        '.right': lambda *args_: "right({})".format(','.join(recurse(x) for x in args_)),
        '.left': lambda *args_: "left({})".format(','.join(recurse(x) for x in args_)),
        '.sentence': "sentence.phrases",

        '.filter_to_tokens': lambda list_: "tokens({})".format(list_),
        '.extract_text': lambda list_: "[p.text.strip() for p in {}]".format(list_),
    }
    # NOTE: only partially complete, many ops still missing
    def recurse(sem):
        if isinstance(sem, tuple):
            if sem[0] in str_ops:
                op = str_ops[sem[0]]
                args_ = [recurse(arg) for arg in sem[1:]]
                return op(*args_) if args_ else op
            else:
                return str(sem)
        else:
            return str(sem)
    return recurse(sem)


core_grammar = GrammarMixin(
    rules=rules,
    ops=ops,
    helpers=helpers,
    annotators=annotators
)
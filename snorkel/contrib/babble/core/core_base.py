from __future__ import print_function

from ..grammar import GrammarMixin, Rule, sems0, sems1, sems_in_order, sems_reversed, flip_dir
from core_templates import PrimitiveTemplate
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
    [Rule('$AtLeastOne', w, ('.geq', ('.int', 1))) for w in ['a', 'another']] +
    [Rule('$Because', w) for w in ['because', 'since', 'if']] +
    [Rule('$Equals', w, '.eq') for w in ['equal', 'equals', '=', '==', 'same', 'identical', 'exactly']] + 
    [Rule('$NotEquals', w, '.neq') for w in ['different']] + 
    [Rule('$LessThan', w, '.lt') for w in ['less than', 'smaller than', '<']] +
    [Rule('$AtMost', w, '.leq') for w in ['at most', 'no larger than', 'less than or equal', 'within', 'no more than', '<=']] +
    [Rule('$AtLeast', w, '.geq') for w in ['at least', 'no less than', 'no smaller than', 'greater than or equal', '>=']] +
    [Rule('$MoreThan', w, '.gt') for w in ['more than', 'greater than', 'larger than', '>']] + 
    [Rule('$In', w, '.in') for w in ['?is in']] +
    [Rule('$Contains', w, '.contains') for w in ['contains', 'contain', 'containing', 'include', 'includes', 'says', 'states', 'mentions', 'mentioned', 'referred', 'refers']] +
    [Rule('$Separator', w) for w in [',', ';', '/']] +
    [Rule('$Possessive', w) for w in ["'s"]] +
    [Rule('$Count', w, '.count') for w in ['number', 'length', 'count']] +
    [Rule('$Punctuation', w) for w in ['.', ',', ';', '!', '?']] +
    [Rule('$Tuple', w, '.tuple') for w in ['pair', 'tuple']] +
    [Rule('$CID', w, '.cid') for w in ['cid', 'cids', 'canonical id', 'canonical ids']] +
    [Rule('$ArgNum', w, ('.int', 1)) for w in ['one', '1']] +
    [Rule('$ArgNum', w, ('.int', 2)) for w in ['two', '2']] +
    [Rule('$ArgXListAnd', w, ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))) for w in ['them']]
)

unary_rules = [
    Rule('$Bool', '$BoolLit', sems0),
    Rule('$BoolLit', '$True', sems0),
    Rule('$BoolLit', '$False', sems0),
    Rule('$Num', '$Int', sems0),
    Rule('$Num', '$Float', sems0),
    Rule('$Conj', '$And', sems0),
    Rule('$Conj', '$Or', sems0),
    Rule('$Exists', '$Is'),
    Rule('$Equals', '$Is ?$Equals', '.eq'),
    Rule('$NotEquals', '$Equals $Not', '.neq'),
    Rule('$Compare', '$Equals', sems0),
    Rule('$Compare', '$NotEquals', sems0),
    Rule('$Compare', '$LessThan', sems0),
    Rule('$Compare', '$AtMost', sems0),
    Rule('$Compare', '$MoreThan', sems0),
    Rule('$Compare', '$AtLeast', sems0),
    Rule('$NumBinToBool', '$Compare', sems0),
    Rule('$NumToBool', '$AtLeastOne', sems0),
]

compositional_rules = [
    ### Top Level ###
    Rule('$ROOT', '$Start $LF $Stop', lambda sems: ('.root', sems[1])),
    Rule('$LF', '$Label $Bool $Because $Bool ?$Punctuation', lambda sems: (sems[0], sems[1], sems[3])),

    ### Logicals ###
    Rule('$Bool', '$Bool $Conj $Bool', lambda sems: (sems[1], sems[0], sems[2])),
    Rule('$Bool', '$Not $Bool', sems_in_order),
    Rule('$Bool', '$All $BoolList', sems_in_order),
    Rule('$Bool', '$Any $BoolList', sems_in_order),
    Rule('$Bool', '$None $BoolList', sems_in_order),

    ### Grouping ###
    Rule('$Bool', '$OpenParen $Bool $CloseParen', lambda (open_, bool_, close_): bool_),

    ### BoolLists ###
    # DEPRECATED:
        # "to the left of arg 2 is a spouse word"
    # Rule('$Bool', '$BoolList', lambda (boollist_,): ('.any', boollist_)),
        # "more than five of X words are upper"
    Rule('$Bool', '$NumToBool $BoolList', lambda (func_,boollist_): ('.call', func_, ('.sum', boollist_))),

    ### Context ###
    Rule('$ArgX', '$Arg $ArgNum', sems_in_order),
    Rule('$ArgXListAnd', '$ArgX $And $ArgX', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))),
    Rule('$ArgXListOr', '$ArgX $Or $ArgX', ('.list', ('.arg', ('.int', 1)), ('.arg', ('.int', 2)))),
]

# template_rules = []
template_rules = (
    PrimitiveTemplate('$ArgX') +
    PrimitiveTemplate('$Num')
)

rules = lexical_rules + unary_rules + compositional_rules + template_rules

ops = {
    # root
    '.root': lambda x: lambda c: x(c),
    '.label': lambda x, y: lambda c: (1 if x(c)==True else -1) if y(c)==True else 0,
    # primitives
    '.bool': lambda x: lambda c: x,
    '.string': lambda x: lambda c: x.encode('utf-8'),
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
    # lists
    '.in': lambda x: lambda cx: lambda y: lambda cy: y(cy) in x(cx),
    '.contains': lambda x: lambda cx: lambda y: lambda cy: x(cx) in y(cy),
    '.count': lambda x: lambda c: len(x(c)),
    '.sum': lambda x: lambda c: sum(x(c)),
    '.intersection': lambda x, y: lambda c: list(set(x(c)).intersection(y(c))),
    # context
    '.arg': lambda x: lambda c: c['candidate'][x(c) - 1],
    }


translate_ops = {
    '.root': lambda LF: LF,
    '.label': lambda label, cond: "return {} if {} else 0".format(1 if label else -1, cond),
    '.bool': lambda bool_: bool_=='True',
    '.string': lambda str_: "'{}'".format(str_),
    '.int': lambda int_: int(int_),
    
    '.tuple': lambda list_: "tuple({})".format(list_),
    '.list': lambda *elements: "[{}]".format(','.join(x.encode('utf-8') for x in elements)),
    '.user_list': lambda name: "${}$".format(name.encode('utf-8')),
    '.map': lambda func_, list_: "map({}, {})".format(func_, list_),
    '.call': lambda func_, args_: "call({}, {})".format(func_, args_),

    '.and': lambda x, y: "({} and {})".format(x, y),
    '.or': lambda x, y: "({} or {})".format(x, y),
    '.not': lambda x: "not ({})".format(x),
    '.all': lambda x: "all({})".format(x),
    '.any': lambda x: "any({})".format(x),
    '.none': lambda x: "not any({})".format(x),

    # '.composite_and': lambda func_, list_: "all(map({}, {}))".format(func_, list_),
    # '.composite_or':  lambda x, y, z: lambda cz: any([x(lambda c: yi)(cxy)(z)(cz)==True for yi in y(cxy)]),

    '.lt': lambda x: "(< {})".format(x),
    '.leq': lambda x: "(<= {})".format(x),
    '.eq': lambda x: "(= {})".format(x),
    '.geq': lambda x: "(>= {})".format(x),
    '.gt': lambda x: "(> {})".format(x),

    '.arg': lambda int_: "arg{}".format(int_),
    '.arg_to_string': lambda arg_: "text({})".format(arg_),
    '.cid': lambda arg_: "cid({})".format(arg_),

    '.in': lambda rhs: "in {}".format(rhs),
    '.contains': lambda rhs: "contains({})".format(rhs),
    '.count': lambda list_: "count({})".format(list_),
    '.sum': lambda arg_: "sum({})".format(arg_),
}


core_grammar = GrammarMixin(
    rules=rules,
    ops=ops,
    helpers={},
    annotators=annotators,
    translate_ops=translate_ops
)
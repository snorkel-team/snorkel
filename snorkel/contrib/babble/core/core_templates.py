from __future__ import print_function

from ..grammar import Rule, sems0, sems1, sems_in_order, sems_reversed


def PrimitiveTemplate(seeds):
    seeds = seeds if isinstance(seeds, list) else [seeds]
    rules = []
    for seed in seeds:
        X = seed
        XListStub = X + 'ListStub'
        XListAnd = X + 'ListAnd'
        XListOr = X + 'ListOr'
        XList = X + 'List'
        XToBool = X + 'ToBool'
        rules.extend([
            # Equality
            # Rule(XToBool, ('$Equals', X))
            
            # To Bool
                # "'a' is uppercase"
                # "[there is an] uppercase 'a'"
            Rule('$Bool', (X, XToBool), lambda (x, func_): ('.call', func_, x)),
            Rule('$Bool', (XToBool, X), lambda (func_, x): ('.call', func_, x)),

            # Not
                # "'a' is not uppercase"
            Rule('$Bool', (X, '$Not', XToBool), lambda (x, not_, func_): (not_, ('.call', func_, x))),

            # Possessive
            # TODO: write test and write rule

            # Building lists
            Rule(XListStub, (X, '?$Separator', X), lambda sems: ('.list', sems[0], sems[2])),
            Rule(XListStub, (XListStub, '?$Separator', X), lambda sems: tuple((list(sems[0]) + [sems[2]]))),

            Rule(XListOr, (X, '?$Separator', '$Or', X), lambda sems: ('.list', sems[0], sems[3])),
            Rule(XListOr, (XListStub, '?$Separator', '$Or', X), lambda sems: tuple(list(sems[0]) + [sems[3]])),
            Rule(XListOr, ('$OpenParen', XListStub, '$CloseParen'), sems1),

            Rule(XListAnd, (X, '?$Separator', '$And', X), lambda sems: ('.list', sems[0], sems[3])),
            Rule(XListAnd, (XListStub, '?$Separator', '$And', X), lambda sems: tuple(list(sems[0]) + [sems[3]])),
            Rule(XListAnd, ('$OpenParen', XListStub, '$CloseParen'), sems1),

            # Generalizing Lists
            Rule(XList, XListStub, sems0),
            Rule(XList, XListAnd, sems0),
            Rule(XList, XListOr, sems0),
            Rule('$List', XList, sems0),

            # Applying functions to lists (normal and inverted order)
                # "'a' or 'b' is in the sentence"
            Rule('$Bool', (XListOr, XToBool), lambda (list_, func_): ('.any', ('.map', func_, list_))),
            Rule('$Bool', (XToBool, XListOr), lambda (func_, list_): ('.any', ('.map', func_, list_))),
                # "'a' and 'b' are in the sentence"
            Rule('$Bool', (XListAnd, XToBool), lambda (list_, func_): ('.all', ('.map', func_, list_))),
            Rule('$Bool', (XToBool, XListAnd), lambda (func_, list_): ('.all', ('.map', func_, list_))),
                # "[at least two of] ('a','b','c') are in the sentence"
            Rule('$BoolList', (XList, XToBool), lambda (list_, func_): ('.map', func_, list_)),
            Rule('$BoolList', (XToBool, XList), lambda (func_, list_): ('.map', func_, list_)),
                # "there is a spouse word in the sentence"
                # "a spouse word is in the sentence"
            Rule('$Bool', ('$Exists', XList, XToBool), lambda (exists_, list_, func_): ('.any', ('.map', func_, list_))),
            Rule('$Bool', (XList, '$Exists', XToBool), lambda (list_, exists_, func_): ('.any', ('.map', func_, list_))),

            # Membership in lists
            Rule(XToBool, ('$In', '$List'), sems_in_order),
            Rule('$Bool', ('$List', '$Contains', X), lambda (list_, contains_, x): (contains_, list_, x)),
        ])
    
    return rules


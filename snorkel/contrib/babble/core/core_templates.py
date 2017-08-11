from __future__ import print_function

from ..grammar import Rule, sems0, sems1, sems_in_order, sems_reversed


def PrimitiveTemplate(seed):
    X = seed
    XListStub = X + 'ListStub'
    XListAnd = X + 'ListAnd'
    XListOr = X + 'ListOr'
    XList = X + 'List'
    XToBool = X + 'ToBool' # f(X) = Bool
    XBinToBool = X + 'BinToBool' # f(X1, X2) = Bool
        
    rules = [
        # Equality
        Rule(XBinToBool, '$Equals', sems0),
        
        # To Bool
            # "'a' is uppercase"
        Rule('$Bool', (X, XToBool), lambda (x, func_): ('.call', func_, x)),
        
        # DEPRECATED:
        # (NOTE: this would introduce false parse for every non-transitive function)
            # "[there is an] uppercase 'a'" 
        # Rule('$Bool', (XToBool, X), lambda (func_, x): ('.call', func_, x)),

            # Case 1: (X (f X))
        Rule(XToBool, (XBinToBool, X), sems_in_order),
            # Case 2: (XList (f X)) - handled naturally with XList XToBool rules
            # Case 3: X f XList
            # TODO: replace '.composite_' functions with fully spelled out versions
        # Rule(Bool, (X, XBinToBool, XListAnd), lambda (x, func_, list_): ('.all', ('.call', (), X)))
        # Rule(Bool, (X, XBinToBool, XListOr), ...)
        Rule(XToBool, (XBinToBool, XListAnd), lambda sems: ('.composite_and', (sems[0],), sems[1])),
        Rule(XToBool, (XBinToBool, XListOr), lambda sems: ('.composite_or', (sems[0],), sems[1])),
            # Case 4: XList (f XList) - handled naturally right now

        # Not
            # "'a' is not uppercase"
        Rule('$Bool', (X, '$Not', XToBool), lambda (x, not_, func_): (not_, ('.call', func_, x))),

        # Possessive
        # TODO: write test and write rule

        # Building lists
        Rule(XListStub, (X, '?$Separator', X), lambda sems: ('.list', sems[0], sems[2])),
        Rule(XListStub, (XListStub, '?$Separator', X), lambda sems: tuple((list(sems[0]) + [sems[2]]))),
        Rule(XList, ('$OpenParen', XListStub, '$CloseParen'), sems1),

        Rule(XListOr, (X, '?$Separator', '$Or', X), lambda sems: ('.list', sems[0], sems[3])),
        Rule(XListOr, (XListStub, '?$Separator', '$Or', X), lambda sems: tuple(list(sems[0]) + [sems[3]])),

        Rule(XListAnd, (X, '?$Separator', '$And', X), lambda sems: ('.list', sems[0], sems[3])),
        Rule(XListAnd, (XListStub, '?$Separator', '$And', X), lambda sems: tuple(list(sems[0]) + [sems[3]])),

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
        Rule('$Bool', ('$Exists', XList, XToBool), lambda (exists_, list_, func_): ('.any', ('.map', func_, list_))),
        
        # DEPRECATED:
            # "a spouse word is in the sentence"
        # Rule('$Bool', (XList, '$Exists', XToBool), lambda (list_, exists_, func_): ('.any', ('.map', func_, list_))),

        # Membership in lists
        Rule(XToBool, ('$In', '$List'), sems_in_order),
        # NOTE: $Contains is still somewhat limited in its functionality
        Rule('$Bool', ('$List', '$Contains', X), lambda (list_, contains_, x): ('.call', ('.in', list_), x)),
        Rule('$Bool', ('$List', '$Contains', XListAnd), lambda (list_, contains_, andlist_): ('.all', ('.map', ('.in', list_), andlist_))),
        Rule('$Bool', ('$List', '$Contains', XListOr), lambda (list_, contains_, orlist_): ('.any', ('.map', ('.in', list_), orlist_))),
    ]
    
    return rules


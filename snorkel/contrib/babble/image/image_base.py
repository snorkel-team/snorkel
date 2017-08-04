from ..grammar import GrammarMixin, Rule, sems0, sems1, sems_in_order, sems_reversed, flip_dir
from image_helpers import helpers

lexical_rules = (
    # Box features
    [Rule('$X', w, ('.int', 0)) for w in ['x']] +
    [Rule('$Y', w, ('.int', 1)) for w in ['y']] +
    [Rule('$Box', w, '.box') for w in ['box']] +

    [Rule('$Top', w, ('.string', 'top')) for w in ['top', 'upper']] +
    [Rule('$Bottom', w, ('.string', 'bottom')) for w in ['bottom', 'lower']] +
    [Rule('$Left', w, ('.string', 'left')) for w in ['left']] +
    [Rule('$Right', w, ('.string', 'right')) for w in ['right']] +
    [Rule('$Below', w, '.below') for w in ['below', 'under']]
)

unary_rules = [
    Rule('$BoxId', '$X', sems0),
    Rule('$BoxId', '$Y', sems0),
    Rule('$Edge', '$Top', sems0),
    Rule('$Edge', '$Bottom', sems0),
    Rule('$Edge', '$Left', sems0),
    Rule('$Edge', '$Right', sems0),
    Rule('$DirectionCompare', '$Below', sems0),
]
    
compositional_rules = [
    Rule('$Bbox', '$Box $BoxId', sems_in_order),
    Rule('$Float', '$Edge $Bbox', lambda (edge, bbox): ('.edge', bbox, edge)),
    
    Rule('$Bool', '$Float $DirectionCompare $Float', lambda (f1, cmp, f2): (cmp, f1, f2)),

    Rule('$BboxToBool', '$Equals $Bbox', sems_in_order),
    Rule('$BboxToBool', '$NotEquals $Bbox', sems_in_order),
    Rule('$Bool', '$Bbox $BboxToBool', lambda (bbox, func_): ('.call', func_, bbox)),
]

rules = lexical_rules + unary_rules + compositional_rules

ops = {
    '.box': lambda int_: lambda c: c['candidate'][int_(c)],
    '.edge': lambda bbox_, side_: lambda c: c['helpers']['extract_edge'](bbox_(c), side_(c)),
    '.below': lambda f1, f2: lambda c: c['helpers']['is_below'](f1(c), f2(c)),
}

image_grammar = GrammarMixin(
    rules=rules,
    ops=ops,
    helpers=helpers,
    annotators=[]
)
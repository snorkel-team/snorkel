from ..grammar import GrammarMixin, Rule, sems0, sems1, sems_in_order, sems_reversed, flip_dir
from image_helpers import helpers

lexical_rules = (
    [Rule('$Box', w, '.box') for w in ['box']] +
    [Rule('$Edge', w, '.edge') for w in ['edge', 'side']] +
    [Rule('$Bottom', w, ('.string', 'bottom')) for w in ['bottom', 'lower']] +
    [Rule('$Top', w, ('.string', 'top')) for w in ['top', 'upper']] +
    [Rule('$Below', w, '.below') for w in ['below', 'under']]
)

unary_rules = [
    Rule('$Side', '$Bottom', sems0),
    Rule('$Side', '$Top', sems0),
    Rule('$DirectionCompare', '$Below', sems0),
]
    
compositional_rules = [
    Rule('$Bbox', '$Box $Int', sems_in_order),
    Rule('$Float', '$Side $Edge $Bbox', lambda (side, edge, bbox): (edge, bbox, side)),
    Rule('$Float', '$Float $Edge $Bbox', lambda (side, edge, bbox): (edge, bbox, side)),
    Rule('$Bool', '$Float $DirectionCompare $Float', lambda (f1, cmp, f2): (cmp, f1, f2)),
]

rules = lexical_rules + unary_rules + compositional_rules

ops = {
    '.box': lambda int_: lambda c: getattr(c['candidate'], 'bboxes')[int_(c) - 1],
    '.edge': lambda bbox_, side_: lambda c: c['helpers']['extract_edge'](bbox_(c), side_(c)),
    '.below': lambda f1, f2: lambda c: c['helpers']['is_below'](f1(c), f2(c)),
}

image_grammar = GrammarMixin(
    rules=rules,
    ops=ops,
    helpers=helpers,
    annotators=[]
)
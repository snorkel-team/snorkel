from ..grammar import GrammarMixin, Rule, sems0, sems1, sems_in_order, sems_reversed, flip_dir
from image_helpers import helpers

lexical_rules = (
    # Box features
    [Rule('$X', w, ('.int', 0)) for w in ['x']] +
    [Rule('$Y', w, ('.int', 1)) for w in ['y']] +
    [Rule('$Box', w, '.box') for w in ['box']] +

    [Rule('$Top', w, ('.string', 'top')) for w in ['top', 'upper', 'uppermost', 'highest']] +
    [Rule('$Bottom', w, ('.string', 'bottom')) for w in ['bottom', 'lower', 'lowest']] +
    [Rule('$Left', w, ('.string', 'left')) for w in ['left']] +
    [Rule('$Right', w, ('.string', 'right')) for w in ['right']] +

    [Rule('$Below', w, '.below') for w in ['below', 'under']] +
    [Rule('$Above', w, '.above') for w in ['above', 'on top of']] +

    [Rule('$Center', w, '.center') for w in ['center', 'middle']] +
    [Rule('$Corner', w, '.corner') for w in ['corner']]
)

unary_rules = [
    Rule('$BoxId', '$X', sems0),
    Rule('$BoxId', '$Y', sems0),
    Rule('$Side', '$Top', sems0),
    Rule('$Side', '$Bottom', sems0),
    Rule('$Side', '$Left', sems0),
    Rule('$Side', '$Right', sems0),

    Rule('$Point', '$Center', sems0),
    Rule('$Point', '$Corner', sems0),
    Rule('$Geometry', '$Point', sems0),
    Rule('$Geometry', '$Edge', sems0),

    Rule('$DirectionCompare', '$Below', sems0),
    Rule('$DirectionCompare', '$Above', sems0),
]
    
compositional_rules = [
    Rule('$Bbox', '$Box $BoxId', sems_in_order),
    Rule('$Edge', '$Side $Bbox', lambda (edge, bbox): ('.edge', bbox, edge)),
    
    Rule('$GeometryToBool', '$DirectionCompare $Geometry', sems_in_order),
    Rule('$Bool', '$Geometry $GeometryToBool', lambda (geom, cmp): ('.call', cmp, geom)),

    Rule('$BboxToBool', '$Equals $Bbox', sems_in_order),
    Rule('$BboxToBool', '$NotEquals $Bbox', sems_in_order),
    Rule('$Bool', '$Bbox $BboxToBool', lambda (bbox, func_): ('.call', func_, bbox)),
]

rules = lexical_rules + unary_rules + compositional_rules

ops = {
    '.box': lambda int_: lambda c: c['candidate'][int_(c)],
    '.edge': lambda bbox_, side_: lambda c: c['helpers']['extract_edge'](bbox_(c), side_(c)),
    '.center': lambda bbox_: lambda c: c['helpers']['extract_center'](bbox_(c)),
    '.corner': lambda bbox_, s1, s2: lambda c: c['helpers']['extract_center'](bbox_(c), s1(c), s2(c)),


    '.below': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_below'](g1(c1), g2(c2)),
}

image_grammar = GrammarMixin(
    rules=rules,
    ops=ops,
    helpers=helpers,
    annotators=[]
)
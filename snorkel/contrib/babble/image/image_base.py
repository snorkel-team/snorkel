from ..grammar import GrammarMixin, Rule, sems0, sems1, sems_in_order, sems_reversed, flip_dir
from image_helpers import helpers

lexical_rules = (
    # Box features
    [Rule('$X', w, ('.int', 0)) for w in ['x']] +
    [Rule('$Y', w, ('.int', 1)) for w in ['y']] +
    [Rule('$Box', w, '.box') for w in ['box']] +

    [Rule('$TopEdge', w, ('.string', 'top')) for w in ['top', 'upper', 'uppermost', 'highest']] +
    [Rule('$BottomEdge', w, ('.string', 'bottom')) for w in ['bottom', 'lower', 'lowest']] +
    [Rule('$LeftEdge', w, ('.string', 'left')) for w in ['left']] +
    [Rule('$RightEdge', w, ('.string', 'right')) for w in ['right']] +

    [Rule('$Below', w, '.below') for w in ['below', 'under']] +
    [Rule('$Above', w, '.above') for w in ['above', 'on top of']] +
    [Rule('$Left', w, '.left') for w in ['left']] +
    [Rule('$Right', w, '.right') for w in ['right']] +
    [Rule('$Near', w, '.near') for w in ['near', 'nearby', 'close']] +
    [Rule('$Far', w, '.far') for w in ['far', 'distant']] +

    [Rule('$Center', w, '.center') for w in ['center', 'middle']] +
    [Rule('$Corner', w, '.corner') for w in ['corner']]
)

unary_rules = [
    Rule('$BoxId', '$X', sems0),
    Rule('$BoxId', '$Y', sems0),
    Rule('$Side', '$TopEdge', sems0),
    Rule('$Side', '$BottomEdge', sems0),
    Rule('$Side', '$LeftEdge', sems0),
    Rule('$Side', '$RightEdge', sems0),

    Rule('$PointCompare', '$Below', sems0),
    Rule('$PointCompare', '$Above', sems0),
    Rule('$PointCompare', '$Left', sems0),
    Rule('$PointCompare', '$Right', sems0),
    Rule('$PointCompare', '$Near', sems0),
    Rule('$PointCompare', '$Far', sems0),
]
    
compositional_rules = [
    Rule('$Bbox', '$Box $BoxId', sems_in_order),
    Rule('$Point', '$Side $Bbox', lambda (side, bbox): ('.edge', bbox, side)),
    Rule('$Point', '$Center $Bbox', lambda (side, bbox): ('.center', bbox)),
    Rule('$Point', '$Side $Side $Corner $Bbox', lambda (s1, s2, _, bbox): ('.corner', bbox, s1, s2)),
    
    Rule('$PointToBool', '$PointCompare $Point', sems_in_order),
    Rule('$Bool', '$Point $PointToBool', lambda (geom, cmp): ('.call', cmp, geom)),
]

rules = lexical_rules + unary_rules + compositional_rules

ops = {
    '.box': lambda int_: lambda c: c['candidate'][int_(c)],
    
    '.edge': lambda bbox_, side_: lambda c: c['helpers']['extract_edge'](bbox_(c), side_(c)),
    '.center': lambda bbox_: lambda c: c['helpers']['extract_center'](bbox_(c)),
    '.corner': lambda bbox_, horz, vert: lambda c: c['helpers']['extract_corner'](bbox_(c), horz(c), vert(c)),

    '.below': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_below'](g1(c1), g2(c2)),
    '.above': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_above'](g1(c1), g2(c2)),
    '.left': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_left'](g1(c1), g2(c2)),
    '.right': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_right'](g1(c1), g2(c2)),
    '.near': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_near'](g1(c1), g2(c2)),
    '.far': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_far'](g1(c1), g2(c2)),
}

image_grammar = GrammarMixin(
    rules=rules,
    ops=ops,
    helpers=helpers,
    annotators=[]
)
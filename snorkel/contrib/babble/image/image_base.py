from ..grammar import GrammarMixin, Rule, sems0, sems1, sems_in_order, sems_reversed, flip_dir
from image_helpers import helpers

lexical_rules = (
    # Box features
    [Rule('$X', w, ('.int', 0)) for w in ['x', 'x.', 'blue']] +
    [Rule('$Y', w, ('.int', 1)) for w in ['y', 'y.', 'yellow']] +
    [Rule('$Box', w, '.box') for w in ['box']] +

    [Rule('$TopEdge', w, ('.string', 'top')) for w in ['top', 'upper', 'uppermost', 'highest']] +
    [Rule('$BottomEdge', w, ('.string', 'bottom')) for w in ['bottom', 'lower', 'lowest']] +
    [Rule('$LeftEdge', w, ('.string', 'left')) for w in ['left']] +
    [Rule('$RightEdge', w, ('.string', 'right')) for w in ['right']] +

    [Rule('$Center', w, '.center') for w in ['center', 'middle']] +
    [Rule('$Corner', w, '.corner') for w in ['corner']] +

    [Rule('$Below', w, '.below') for w in ['below', 'under', 'underneath', 'lower']] +
    [Rule('$Above', w, '.above') for w in ['above', 'on top of', 'higher']] +
    [Rule('$Left', w, '.left') for w in ['left']] +
    [Rule('$Right', w, '.right') for w in ['right']] +
    [Rule('$Near', w, '.near') for w in ['near', 'nearby', 'close', 'over', 'in', 'same place', 'at', 'even', 'equal']] +
    [Rule('$Far', w, '.far') for w in ['far', 'distant']] +

    [Rule('$Smaller', w, '.smaller') for w in ['smaller', 'tinier', 'fraction', 'half']] +
    [Rule('$Larger', w, '.larger') for w in ['larger', 'bigger', 'big']] +
    
    [Rule('$Wider', w, '.wider') for w in ['wider', 'broader', 'long', 'wide', 'broad']] +
    [Rule('$Taller', w, '.taller') for w in ['taller', 'longer']] +
    [Rule('$Skinnier', w, '.skinnier') for w in ['skinnier', 'slimmer']] +
    [Rule('$Shorter', w, '.shorter') for w in ['shorter']] +
    
    [Rule('$Overlaps', w, '.overlaps') for w in ['overlaps', 'intersects', 'bisects', 'overlapping']] +
    
    [Rule('$Surrounds', w, '.surrounds') for w in ['encloses', 'envelopes', 'surrounds', 'around', 'engulfs', 'includes']] +
    [Rule('$Within', w, '.within') for w in ['within', 'inside', 'enclosed', 'enveloped', 'fits', 'surrounded']] 
    
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

    Rule('$BoxCompare', '$Smaller', sems0),
    Rule('$BoxCompare', '$Larger', sems0),
    
    Rule('$BoxCompare', '$Near', sems0),
    Rule('$BoxCompare', '$Far', sems0),
    
    Rule('$BoxCompare', '$Taller', sems0),
    Rule('$BoxCompare', '$Wider', sems0),
    Rule('$BoxCompare', '$Skinnier', sems0),
    Rule('$BoxCompare', '$Shorter', sems0),
    
    Rule('$BoxCompare', '$Overlaps', sems0),
    
    Rule('$BoxCompare', '$Surrounds', sems0),
    Rule('$BoxCompare', '$Within', sems0),
]
    
compositional_rules = [
    Rule('$Bbox', '$Box $BoxId', sems_in_order),
    Rule('$Point', '$Side $Bbox', lambda (side, bbox): ('.edge', bbox, side)),
    Rule('$Point', '$Center $Bbox', lambda (side, bbox): ('.center', bbox)),
    Rule('$Point', '$Side $Side $Corner $Bbox', lambda (s1, s2, _, bbox): ('.corner', bbox, s1, s2)),
    
    Rule('$PointToBool', '$PointCompare $Point', sems_in_order), # "is below the center of Box X"
    Rule('$PointToBool', '$PointCompare $Bbox', sems_in_order), # "is below Box X (use smart edge choice)"
    Rule('$Bool', '$Point $PointToBool', lambda (point, cmp): ('.call', cmp, point)),

    Rule('$BboxToBool', '$PointCompare $Bbox', sems_in_order), # "is below Box X (use smart edge choice)"
    Rule('$BboxToBool', '$BoxCompare $Bbox', sems_in_order), # "is smaller than Box X"
    Rule('$Bool', '$Bbox $BboxToBool', lambda (point, cmp): ('.call', cmp, point)),
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

    '.smaller': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_smaller'](g1(c1), g2(c2)),
    '.larger': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_larger'](g1(c1), g2(c2)),
    
    '.wider': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_wider'](g1(c1), g2(c2)),
    '.taller': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_taller'](g1(c1), g2(c2)),
    '.skinnier': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_skinnier'](g1(c1), g2(c2)),
    '.shorter': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_shorter'](g1(c1), g2(c2)),
    
    '.overlaps': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_overlaps'](g1(c1), g2(c2)),
    
    '.surrounds': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_surrounds'](g1(c1), g2(c2)),
    '.within': lambda g2: lambda c2: lambda g1: lambda c1: c1['helpers']['is_within'](g1(c1), g2(c2)),
}

image_grammar = GrammarMixin(
    rules=rules,
    ops=ops,
    helpers=helpers,
    annotators=[]
)
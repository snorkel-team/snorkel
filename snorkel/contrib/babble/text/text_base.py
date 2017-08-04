from ..grammar import GrammarMixin, Rule, sems0, sems1, sems_in_order, sems_reversed, flip_dir
from text_helpers import helpers

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
]
    
compositional_rules = [
    # NER/POS
    Rule('$PhraseList', '$POS $PhraseList', lambda sems: ('.filter_by_attr', sems[1], ('.string', 'pos_tags'), sems[0])),
    Rule('$PhraseList', '$NER $PhraseList', lambda sems: ('.filter_by_attr', sems[1], ('.string', 'ner_tags'), sems[0])),
    Rule('$TokenList', '$PhraseList', lambda sems: ('.filter_to_tokens', sems[0])),
    Rule('$StringList', '$PhraseList', lambda sems: ('.extract_text', sems[0])),    
]

rules = lexical_rules + unary_rules + compositional_rules

ops = {
    '.left': lambda *x: lambda cx: cx['helpers']['get_left_phrases'](*[xi(cx) for xi in x]),
    '.right': lambda *x: lambda cx: cx['helpers']['get_right_phrases'](*[xi(cx) for xi in x]),
    '.between': lambda x: lambda c: c['helpers']['get_between_phrases'](*[xi for xi in x(c)]),
    '.sentence': lambda c: c['helpers']['get_sentence_phrases'](c['candidate'][0]),
    '.extract_text': lambda phrlist: lambda c: [getattr(p, 'text').strip() for p in phrlist(c)],
    '.filter_by_attr': lambda phrlist, attr, val: lambda c: [p for p in phrlist(c) if getattr(p, attr(c))[0] == val(c)],
    '.filter_to_tokens': lambda phrlist: lambda c: [p for p in phrlist(c) if len(getattr(p, 'words')) == 1], 
}

text_grammar = GrammarMixin(
    rules=rules,
    ops=ops,
    helpers=helpers,
    annotators=[]
)
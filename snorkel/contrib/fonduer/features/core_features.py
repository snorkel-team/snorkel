from snorkel.models import TemporarySpan, ImplicitSpan

DEF_VALUE = 1


def get_core_feats(candidate):
    args = candidate.get_contexts()
    if not (isinstance(args[0], TemporarySpan)):
        raise ValueError("Accepts Span-type arguments, %s-type found." % type(candidate))

    # Unary candidates
    if len(args) == 1:
        span = args[0]
        for f in _generate_core_feats(span):
            yield 'CORE_' + f, DEF_VALUE
    # Binary candidates
    elif len(args) == 2:
        span1, span2 = args
        for f in _generate_core_feats(span1):
            yield 'CORE_e1_' + f, DEF_VALUE
        for f in _generate_core_feats(span2):
            yield 'CORE_e2_' + f, DEF_VALUE         
    else:
        raise NotImplementedError("Only handles unary and binary candidates currently")


def _generate_core_feats(span):
    yield "SPAN_TYPE_[%s]" % ('IMPLICIT' if isinstance(span, ImplicitSpan) else 'EXPLICIT')

    if span.get_span()[0].isupper():
        yield "STARTS_WITH_CAPITAL"

    yield "LENGTH_{}".format(span.get_n())

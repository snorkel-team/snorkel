from __future__ import unicode_literals
from snorkel.models import TemporarySpan
from ..models import ImplicitSpan


FEAT_PRE = "CORE_"
DEF_VALUE = 1

unary_feats = {}


def get_core_feats(candidates):
    candidates = candidates if isinstance(candidates, list) else [candidates]
    for candidate in candidates:
        args = candidate.get_contexts()
        if not (isinstance(args[0], TemporarySpan)):
            raise ValueError("Accepts Span-type arguments, %s-type found." % type(candidate))

        # Unary candidates
        if len(args) == 1:
            span = args[0]

            if span.stable_id not in unary_feats:
                unary_feats[span.stable_id] = set()
                for f in _generate_core_feats(span):
                    unary_feats[span.stable_id].add(f)

            for f in unary_feats[span.stable_id]:
                yield candidate.id, FEAT_PRE + f, DEF_VALUE

        # Binary candidates
        elif len(args) == 2:
            span1, span2 = args
            for span, pre in [(span1, "e1_"), (span2, "e2_")]:
                if span.stable_id not in unary_feats:
                    unary_feats[span.stable_id] = set()
                    for f in _generate_core_feats(span):
                        unary_feats[span.stable_id].add(f)

                for f in unary_feats[span.stable_id]:
                    yield candidate.id, FEAT_PRE + pre + f, DEF_VALUE
        else:
            raise NotImplementedError("Only handles unary and binary candidates currently")


def _generate_core_feats(span):
    yield "SPAN_TYPE_[%s]" % ('IMPLICIT' if isinstance(span, ImplicitSpan) else 'EXPLICIT')

    if span.get_span()[0].isupper():
        yield "STARTS_WITH_CAPITAL"

    yield "LENGTH_{}".format(span.get_n())

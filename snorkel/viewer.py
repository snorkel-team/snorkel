from __future__ import print_function
try:
    from IPython.core.display import display, Javascript
except:
    raise Exception("This module must be run in IPython.")
from itertools import islice
from random import randint, sample
import os
from collections import defaultdict
import ipywidgets as widgets
from traitlets import Unicode, Int, Dict, List

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

HOME = os.environ['SNORKELHOME']


# PAGE LAYOUT TEMPLATES
LI_HTML = """
<li class="list-group-item" data-toggle="tooltip" data-placement="top" title="{context_id}">{data}</li>
"""

PAGE_HTML = """
<div class="viewer-page" id="viewer-page-{pid}">
    <ul class="list-group">{data}</ul>
</div>
"""

class Viewer(widgets.DOMWidget):
    """
    Generic object for viewing and labeling candidate objects in their rendered contexts.
    Takes in:
        - A list of *contexts* (e.g. Sentence objects) having an id attribute;
        - A list of *candidates( (e.g. Ngram mention spans), having a join attribute fn
            (candidate_join_key_fn) such that contexts and candidates are joined on
            context.id == candidate_join_key_fn(candidate)
        - Optionally: a list of *gold annotations* of the same type as the candidates
        - A max number of contexts to render (n_max)
    By default, contexts with no candidates or gold annotations are filtered out, however
    this can be disabled (filter_empty) and any filtering can be done prior to passing into
    the Viewer object!
    """
    _view_name         = Unicode('ViewerView').tag(sync=True)
    _view_module       = Unicode('viewer').tag(sync=True)
    cids               = List().tag(sync=True)
    html               = Unicode('<h3>Error!</h3>').tag(sync=True)
    _labels_serialized = Unicode().tag(sync=True)
    selected_cid       = Int().tag(sync=True)

    def __init__(self, candidates, gold=[], n_per_page=3, height=225):
        super(Viewer, self).__init__()

        # Viewer display configs
        self.n_per_page = n_per_page
        self.height     = height

        # Get all the contexts containing the candidates
        self.candidates = set(candidates)
        self.gold       = set(gold)

        # TODO: Hack!!!
        try:
            self.contexts = list(set(c.context for c in self.candidates.union(self.gold)))
        except:
            self.contexts = list(set(c.span0.context for c in self.candidates.union(self.gold)))

        # TODO: Replace with proper ORM syntax
        self.candidates_by_id = dict([(c.id, c) for c in self.candidates])

        # TODO: Remove this workaround

        # display js, construct html and pass on to widget model
        self.render()

    def _tag_span(self, html, cids, gold=False):
        """
        Create the span around a segment of the context associated with one or more candidates / gold annotations
        """
        classes  = ['candidate'] if len(cids) > 0 else []
        classes += ['gold-annotation'] if gold else []
        classes += map(str, cids)
        return '<span class="{classes}">{html}</span>'.format(classes=' '.join(classes), html=html)

    def _tag_context(self, context, candidates, gold):
        """Given the raw context, tag the spans using the generic _tag_span method"""
        raise NotImplementedError()

    def render(self):
        """Renders viewer pane"""
        cids = []

        # Iterate over pages of contexts
        pid   = 0
        pages = []
        N     = len(self.contexts)
        for i in range(0, N, self.n_per_page):
            page_cids = []
            lis       = []
            for j in range(i, min(N, i + self.n_per_page)):
                context = self.contexts[j]

                # NOTE: We do *not* assume that the user wants to see all candidates in each context- only
                # the candidates that are passed in
                #candidates = self.candidates.intersection(context.candidates)
                try:
                    candidates = [c for c in self.candidates if c.context == context]
                except:
                    candidates = [c for c in self.candidates if c.span0.context == context]

                # TODO: Replace this (and other similar) statements with SQLAlchemy syntax
                gold = [g for g in self.gold if g.context_id == context.id]

                # Construct the <li> and page view elements
                li_data = self._tag_context(context, candidates, gold)
                lis.append(LI_HTML.format(data=li_data, context_id=context.id))

                # TODO: Remove this hack
                try:
                    page_cids += [c.id for c in sorted(candidates, key=lambda c : c.char_start)]
                except:
                    page_cids += [c.id for c in sorted(candidates, key=lambda c : c.span0.char_start)]
            pages.append(PAGE_HTML.format(pid=pid, data=''.join(lis)))
            cids.append(page_cids)
            pid += 1

        # Render in primary Viewer template
        self.cids = cids
        self.html = open(HOME + '/viewer/viewer.html').read().format(bh=self.height, data=''.join(pages))
        display(Javascript(open(HOME + '/viewer/viewer.js').read()))

    def get_labels(self):
        # TODO: Create an ORM object for labels!!!
        """De-serialize labels, map to candidate id, and return as dictionary"""
        labels = [x.split('~~') for x in self._labels_serialized.split(',') if len(x) > 0]
        LABEL_MAP = {'true':1, 'false':-1}
        return dict([(id, LABEL_MAP.get(l, 0)) for id,l in labels])

    def get_selected(self):
        return self.candidates_by_id[self.selected_cid]


class SentenceNgramViewer(Viewer):
    """Viewer for Sentence objects and candidate Spans within them"""
    def __init__(self, candidates, gold=[], n_per_page=3, height=225):
        super(SentenceNgramViewer, self).__init__(candidates, gold=gold, n_per_page=n_per_page, height=height)

    def _is_subspan(self, s, e, c):
        return s >= c.char_start and e <= c.char_end

    def _tag_context(self, sentence, candidates, gold):
        """Tag **potentially overlapping** spans of text, at the character-level"""
        s = sentence.text

        # First, split the sentence into the *smallest* single-candidate chunks
        # TODO: Remove this hack!
        try:
            both   = list(candidates) + list(gold)
            splits = sorted(list(set([b.char_start for b in both] + [b.char_end + 1 for b in both] + [0, len(s)])))
        except:
            both   = [c.span0 for c in candidates] + [c.span1 for c in candidates] + [g.span0 for g in gold] + [g.span1 for g in gold]
            splits = sorted(list(set([b.char_start for b in both] + [b.char_end + 1 for b in both] + [0, len(s)])))

        # For each chunk, add cid if subset of candidate span, tag if gold, and produce span
        html = ""
        for i in range(len(splits)-1):
            start  = splits[i]
            end    = splits[i+1] - 1

            # TODO: Remove this hack!
            try:
                cids  = [c.id for c in candidates if self._is_subspan(start, end, c)]
                gcids = [g.id for g in gold if self._is_subspan(start, end, g)]
            except:
                cids  = [c.id for c in candidates if self._is_subspan(start, end, c.span0) or self._is_subspan(start, end, c.span1)]
                gcids = [g.id for g in gold if self._is_subspan(start, end, g.span0) or self._is_subspan(start, end, g.span1)]

            html += self._tag_span(s[start:end+1], cids, gold=len(gcids) > 0)
        return html

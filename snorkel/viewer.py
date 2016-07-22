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
    selected_cid       = Unicode().tag(sync=True)

    def __init__(self, contexts, candidates, candidate_join_key_fn, gold=[], n_max=100, filter_empty=True, n_per_page=3, height=225):
        super(Viewer, self).__init__()

        # Viewer display configs
        self.n_per_page = n_per_page
        self.height     = height

        # Index candidates by id
        self.candidates = {}
        for c in candidates:
            self.candidates[c.id] = c

        # Index candidates by context
        candidates_index = defaultdict(list)
        for c in candidates:
            candidates_index[candidate_join_key_fn(c)].append(c)

        # Index gold annotations by context
        gold_index = defaultdict(list)
        for g in gold:
            gold_index[candidate_join_key_fn(g)].append(g)

        # Store as list of (context, candidates, gold) 'views'
        self.views = []
        for c in contexts:
            if len(self.views) == n_max:
                break
            if len(candidates_index[c.id]) + len(gold_index[c.id]) > 0 or not filter_empty:
                self.views.append((c, candidates_index[c.id], gold_index[c.id]))

        # display js, construct html and pass on to widget model
        self.render()

    def _tag_span(self, html, cids, gold=False):
        """Create the span around a segment of the context associated with one or more candidates / gold annotations"""
        classes  = ['candidate'] if len(cids) > 0 else []
        classes += ['gold-annotation'] if gold else []
        classes += cids
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
        N     = len(self.views)
        for i in range(0, N, self.n_per_page):
            pg_cids = []
            lis     = []
            for j in range(i, min(N, i + self.n_per_page)):
                context, candidates, gold = self.views[j]
                li_data = self._tag_context(context, candidates, gold)
                lis.append(LI_HTML.format(data=li_data, context_id=context.id))
                pg_cids += [c.id for c in sorted(candidates, key=lambda c : c.char_start)]
            pages.append(PAGE_HTML.format(pid=pid, data=''.join(lis)))
            cids.append(pg_cids)
            pid += 1

        # Render in primary Viewer template
        self.cids    = cids
        self.html    = open(HOME + '/viewer/viewer.html').read().format(bh=self.height, data=''.join(pages))
        display(Javascript(open(HOME + '/viewer/viewer.js').read()))

    def get_labels(self):
        """De-serialize labels, map to candidate id, and return as dictionary"""
        return dict(x.split('~~') for x in self._labels_serialized.split(',') if len(x) > 0)

    def get_selected(self):
        if len(self.selected_cid) > 0:
            return self.candidates[self.selected_cid]
        else:
            return None


class SentenceNgramViewer(Viewer):
    """Viewer for Sentence objects and Ngram candidate spans within them, given a Corpus object"""
    def __init__(self, sentences, candidates, gold=[], n_max=100, filter_empty=True, n_per_page=3, height=225):
        super(SentenceNgramViewer, self).__init__(sentences, candidates, lambda c : c.sent_id, gold=gold, n_max=n_max, filter_empty=filter_empty, n_per_page=n_per_page, height=height)

    def _is_subspan(self, s, e, c):
        return s >= c.sent_char_start and e <= c.sent_char_end

    def _tag_context(self, sentence, candidates, gold):
        """Tag **potentially overlapping** spans of text, at the character-level"""
        s = sentence.text

        # First, split the sentence into the *smallest* single-candidate chunks
        both   = candidates + gold
        splits = sorted(list(set([b.sent_char_start for b in both] + [b.sent_char_end + 1 for b in both] + [0, len(s)])))

        # For each chunk, add cid if subset of candidate span, tag if gold, and produce span
        html = ""
        for i in range(len(splits)-1):
            start  = splits[i]
            end    = splits[i+1] - 1
            cids   = [c.id for c in candidates if self._is_subspan(start, end, c)]
            gcids  = [g.id for g in gold if self._is_subspan(start, end, g)]
            html += self._tag_span(s[start:end+1], cids, gold=len(gcids) > 0)
        return html

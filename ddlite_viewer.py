from itertools import islice
from random import randint, sample
import os
from collections import defaultdict

HOME = os.environ['DDLHOME']

try:
    from IPython.core.display import display_html, HTML, display_javascript, Javascript
except:
    raise Exception("This module must be run in IPython.")


JS_LIBS = [
    "http://d3js.org/d3.v3.min.js",
    "https://ajax.googleapis.com/ajax/libs/jquery/1.11.3/jquery.min.js",
    #"%s/viewer/bootstrap/js/bootstrap.min.js" % HOME
]

def render(html, js, css_isolated=False):
    """Renders an html + js payload in IPython"""
    display_html(HTML(data=html), metadata=dict(isolated=css_isolated))
    if js is not None:
        display_javascript(Javascript(data=js, lib=JS_LIBS))


class Viewer(object):
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
    def __init__(self, contexts, candidates, candidate_join_key_fn, gold=[], n_max=100, filter_empty=True):
        
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

    def _tag_span(self, html, vid, pid, cids, gold=False):
        """
        Given interior html and a viewer id (vid), page id (pid), and **list of candidate 
        ids (cids) that use this span**, return a the span-enclosed html such that it 
        will be highlighted, browsable and taggable in the Viewer module.
        """
        classes =  []
        if len(cids) > 0:
            classes.append('candidate')

        # For gold annotations, no scrolling, so only need to mark as gold
        if gold:
            classes.append('gold-annotation')

        # Mark each cid
        classes += ['c-{vid}-{pid}-{cid}'.format(vid=vid, pid=pid, cid=c) for c in cids]
        return '<span class="{classes}">{html}</span>'.format(classes=' '.join(classes), html=html)

    def _tag_context(self, context, candidates, gold, vid, pid, cid_offset=0):
        """Given the raw context, tag the spans using the generic _tag_span method"""
        raise NotImplementedError()

    def render(self, n_per_page=3, height=225):
        """Renders viewer pane"""
        N = len(self.views)
        li_html   = """
        <li class="list-group-item" data-toggle="tooltip" data-placement="top" title="{cid}">{data}</li>
        """
        page_html = """
        <div class="viewer-page viewer-page-{vid}" id="viewer-page-{vid}-{pid}" data-nc="{nc}">
            <ul class="list-group">{data}</ul>
        </div>
        """

        # Random viewer id to avoid js cross-cell collisions
        vid = randint(0,10000)

        # Iterate over pages of contexts
        pid   = 0
        pages = []
        for i in range(0, N, n_per_page):
            lis        = []
            cid_offset = 0
            for j in range(i, min(N, i+n_per_page)):
                context, candidates, gold = self.views[j]
                li_data = self._tag_context(context, candidates, gold, vid, pid, cid_offset=cid_offset)
                lis.append(li_html.format(data=li_data, cid=context.id))
                cid_offset += len(candidates)
            pages.append(page_html.format(vid=vid, pid=pid, nc=cid_offset, data=''.join(lis)))
            pid += 1

        # Render in primary Viewer template
        html = open(HOME + '/viewer/viewer.html').read().format(vid=vid, bh=height, data=''.join(pages))
        js   = open(HOME + '/viewer/viewer.js').read() % (vid, len(pages))
        render(html, js)


class SentenceNgramViewer(Viewer):
    """Viewer for Sentence objects and Ngram candidate spans within them, given a Corpus object"""

    def _tag_context(self, sentence, candidates, gold, vid, pid, cid_offset=0):
        """Tag **potentially overlapping** spans of text, at the character-level"""
        context_html = sentence.text

        # Sort candidates by char_start
        candidates.sort(key=lambda c : c.char_start)
        gold.sort(key=lambda g : g.char_start)

        # First, we split the string up into chunks by unioning all span start / end points
        splits  = [c.sent_char_start for c in candidates] + [c.sent_char_end + 1 for c in candidates]
        splits += [g.sent_char_start for g in gold] + [g.sent_char_end + 1 for g in gold]
        if len(splits) == 0:
            return context_html
        splits  = sorted(list(set(splits)))
        splits  = splits if splits[0] == 0 else [0] + splits

        # Tag by classes
        span_cids    = defaultdict(list)
        span_is_gold = defaultdict(bool)
        for i,c in enumerate(candidates):
            for j in range(splits.index(c.sent_char_start), splits.index(c.sent_char_end+1)):
                span_cids[splits[j]].append(i + cid_offset)
        for i,g in enumerate(gold):
            for j in range(splits.index(g.sent_char_start), splits.index(g.sent_char_end+1)):
                span_is_gold[splits[j]] = True

        # Also include candidate metadata- as hidden divs
        # TODO: Handle this in nicer way!
        html = ""
        for i,c in enumerate(candidates):

            # Set the caption shown when candidate is highlighted
            cap   = "CID: %s" % c.id
            html += '<div class="candidate-data" id="cdata-{vid}-{pid}-{cid}" caption="{cap}"></div>'.format(vid=vid, pid=pid, cid=i+cid_offset, cap=cap)

        # Render as sequence of spans
        for i,s in enumerate(splits):
            end   = splits[i+1] if i < len(splits)-1 else len(context_html)
            html += self._tag_span(context_html[s:end], vid, pid, span_cids[s], gold=span_is_gold[s])
        return html

from itertools import islice
from random import randint
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
    Generic object for viewing and labeling candidate objects in their rendered context
    Takes a list of *views* which are (context, candidates) tuples, where candidates is a list of candidate objects
    """
    def __init__(self, corpus):
        self.views = self._corpus_generator(corpus)

    def _corpus_generator(self, corpus):
        """Access the corpus objects generator method to yield (context, candidates) tuples"""
        return corpus

    def _tag_span(self, html, vid, pid, cids):
        """
        Given interior html and a viewer id (vid), page id (pid), and **list of candidate 
        ids (cids) that use this span**, return a the span-enclosed html such that it 
        will be highlighted, browsable and taggable in the Viewer module.
        """
        classes =  []
        if len(cids) > 0:
            classes.append('candidate')
        classes += ['c-{vid}-{pid}-{cid}'.format(vid=vid, pid=pid, cid=c) for c in cids]
        return '<span class="{classes}">{html}</span>'.format(classes=' '.join(classes), html=html)

    def _tag_context(self, context, candidates, vid, pid, cid_offset=0):
        """Given the raw context, tag the spans using the generic _tag_span method"""
        raise NotImplementedError()

    def render(self, n=24, n_per_page=3, body_height_px=250):
        """Renders viewer pane"""

        # Random viewer id to avoid js cross-cell collisions
        vid = randint(0,10000)

        # Render the generic html
        li_html   = '<li class="list-group-item">{data}</li>'
        page_html = """
            <div class="viewer-page viewer-page-{vid}" id="viewer-page-{vid}-{pid}" data-nc="{nc}">
                <ul class="list-group">{data}</ul>
            </div>
            """
        
        # Iterate over pages of contexts
        # TODO: Don't materialize the full list of all candidates here!
        views = list(self.views)
        pid   = 0
        pages = []
        for i in range(0, n, n_per_page):
            lis        = []
            cid_offset = 0
            for j in range(i, i+n_per_page):
                context, candidates = views[j]
                lis.append(li_html.format(data=self._tag_context(context, candidates, vid, pid, cid_offset=cid_offset)))
                cid_offset += len(candidates)
            pages.append(page_html.format(vid=vid, pid=pid, nc=cid_offset, data=''.join(lis)))
            pid += 1

        # Render in primary Viewer template
        html = open(HOME + '/viewer/viewer.html').read().format(vid=vid, bh=body_height_px, data=''.join(pages))
        js   = open(HOME + '/viewer/viewer.js').read() % (vid, len(pages))
        render(html, js)


class SentenceNgramViewer(Viewer):
    """Viewer for Sentence objects and Ngram candidate spans within them, given a Corpus object"""
    def _corpus_generator(self, corpus):
        return corpus.iter_sentences_and_candidates()

    def _tag_context(self, context, candidates, vid, pid, cid_offset=0):
        """Tag **potentially overlapping** spans of text, at the character-level"""
        context_html = context.text
        if len(candidates) == 0:
            return context_html

        # First, we split the string up into chunks by unioning all span start / end points
        splits = sorted(list(set([c.sent_char_start for c in candidates] + [c.sent_char_end + 1 for c in candidates])))
        splits = splits if splits[0] == 0 else [0] + splits

        # Tag by classes
        span_cids = defaultdict(list)
        for i,c in enumerate(candidates):
            for j in range(splits.index(c.sent_char_start), splits.index(c.sent_char_end+1)):
                span_cids[splits[j]].append(i + cid_offset)

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
            html += self._tag_span(context_html[s:end], vid, pid, span_cids[s])
        return html

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
    "viewer/bootstrap/js/bootstrap.min.js"
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

    def _get_html(self, context):
        """Renders a context to basic html form"""
        return context

    def _tag_html(self, context_html, candidates):
        """Tags candidates within html context in spans, with ids corresponding to candidate.id"""
        return context_html
    
    def render(self, n=5):
        """Renders viewer pane"""

        # Random viewer id to avoid js cross-cell collisions
        vid = randint(0,10000)

        # Render the generic html
        lis     = []
        li_html = '<li class="list-group-item" id="%s-%s">%s</li>'
        for i,c in enumerate(islice(self.views, n)):
            context, candidates = c

            # Render each context as a separate div
            lis.append(li_html % (vid, i, self._tag_html(self._get_html(context), candidates)))

        # Render in primary Viewer template
        html = open(HOME + '/viewer/viewer.html').read().format(vid=vid, data="\n".join(lis))
        render(html, None)


class SentenceViewer(Viewer):
    """Viewer for Sentence objects and candidate spans within them, given a Corpus object"""
    def _corpus_generator(self, corpus):
        return corpus.iter_sentences_and_candidates()

    def _get_html(self, context):
        return context.text

    def _tag_html(self, context_html, candidates):
        """Tag **potentially overlapping** spans of text, at the character-level"""
        if len(candidates) == 0:
            return context_html

        # First, we split the string up into chunks by unioning all span start / end points
        splits = sorted(list(set([c.char_start for c in candidates] + [c.char_end + 1 for c in candidates])))
        splits = splits if splits[0] == 0 else [0] + splits

        # Tag by classes
        span_classes = defaultdict(list)
        for c in candidates:
            for i in range(splits.index(c.char_start), splits.index(c.char_end+1)):
                span_classes[splits[i]].append("yellow")  # TODO: Make this class candidate-specific!

        # Render as sequence of spans
        html = ""
        for i,s in enumerate(splits):
            end   = splits[i+1] if i < len(splits)-1 else len(context_html)
            html += '<span class="%s">%s</span>' % (' '.join(span_classes[s]), context_html[s:end])  # TODO: Add id!
        return html

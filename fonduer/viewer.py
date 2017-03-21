from __future__ import print_function
from .models import Label
from .queries import get_or_create_single_key_set
try:
    from IPython.core.display import display, Javascript
except:
    raise Exception("This module must be run in IPython.")
import os
import ipywidgets as widgets
from traitlets import Unicode, Int, List
import getpass
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

HOME = os.environ['SNORKELHOME']


# PAGE LAYOUT TEMPLATES
LI_HTML = u"""
<li class="list-group-item" data-toggle="tooltip" data-placement="top" title="{context_id}">{data}</li>
"""

PAGE_HTML = u"""
<div class="viewer-page" id="viewer-page-{pid}">
    <ul class="list-group">{data}</ul>
</div>
"""


class Viewer(widgets.DOMWidget):
    """
    Generic object for viewing and labeling Candidate objects in their rendered Contexts.
    """
    _view_name         = Unicode('ViewerView').tag(sync=True)
    _view_module       = Unicode('viewer').tag(sync=True)
    cids               = List().tag(sync=True)
    html               = Unicode('<h3>Error!</h3>').tag(sync=True)
    _labels_serialized = Unicode().tag(sync=True)
    _selected_cid      = Int().tag(sync=True)

    def __init__(self, candidates, session, gold=[], n_per_page=3, height=225, annotator_name=None):
        """
        Initializes a Viewer.

        The Viewer uses the keyword argument annotator_name to define an AnnotationKeySet with that name, containing
        a single AnnotationKey of the same name. If it already exists, it will be resused, but if the AnnotationKeySet
        already exists and contains other keys, the Viewer will raise an error.

        This AnnotationKeySet can be used to retrieve the labels with a LabelManager, and corresponding AnnotationKeys
        can be grouped into a new AnnotationKeySet to manage the work of multiple annotators simultaneously.

        :param candidates: A Python container of Candidates (e.g., not a CandidateSet, but candidate_set.candidates)
        :param session: The SnorkelSession for the database backend
        :param gold: Optional, Python container of Candidates that are know to have positive labels
        :param n_per_page: Optional, number of Contexts to display per page
        :param height: Optional, the height in pixels of the Viewer
        :param annotator_name: Name of the human using the Viewer, for saving their work. Defaults to system username.
        """
        super(Viewer, self).__init__()
        self.session = session

        # By default, use the username as annotator name
        name = annotator_name if annotator_name is not None else getpass.getuser()

        # Sets up the AnnotationKey to use
        try:
            _, self.annotator = get_or_create_single_key_set(self.session, name)
        except ValueError:
            raise ValueError('annotator_name ' + unicode(name) + ' is already in use for an incompatible ' +
                             'AnnotationKey and/or AnnotationKeySet. Please specify a new annotator_name.')

        # Viewer display configs
        self.n_per_page = n_per_page
        self.height     = height

        # Note that the candidates are not necessarily commited to the DB, so they *may not have* non-null ids
        # Hence, we index by their position in this list
        # We get the sorted candidates and all contexts required, either from unary or binary candidates
        self.gold = list(gold)
        self.candidates = sorted(list(candidates), key=lambda c : c[0].char_start)
        self.contexts   = list(set(c[0].parent for c in self.candidates + self.gold))
        
        # If committed, sort contexts by id
        try:
            self.contexts = sorted(self.contexts, key=lambda c : c.id)
        except:
            pass

        # Loads existing annotations
        self.annotations = [None] * len(self.candidates)
        init_labels_serialized = []
        for i, candidate in enumerate(self.candidates):
            existing_annotation = self.session.query(Label) \
                .filter(Label.key == self.annotator) \
                .filter(Label.candidate == candidate) \
                .first()
            if existing_annotation is not None:
                self.annotations[i] = existing_annotation
                if existing_annotation.value == 1:
                    value_string = 'true'
                elif existing_annotation.value == -1:
                    value_string = 'false'
                else:
                    raise ValueError(str(existing_annotation) +
                                     ' has value not in {1, -1}, which Viewer does not support.')
                init_labels_serialized.append(str(i) + '~~' + value_string)
        self._labels_serialized = ','.join(init_labels_serialized)

        # Configures message handler
        self.on_msg(self.handle_label_event)

        # display js, construct html and pass on to widget model
        self.render()

    def _tag_span(self, html, cids, gold=False):
        """
        Create the span around a segment of the context associated with one or more candidates / gold annotations
        """
        classes  = ['candidate'] if len(cids) > 0 else []
        classes += ['gold-annotation'] if gold else []
        classes += map(str, cids)

        # Scrub for non-ascii characters; replace with ?
        return u'<span class="{classes}">{html}</span>'.format(classes=' '.join(classes), html=html)

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

                # Get the candidates in this context
                candidates = [c for c in self.candidates if c[0].parent == context]
                gold = [g for g in self.gold if g.context_id == context.id]

                # Construct the <li> and page view elements
                li_data = self._tag_context(context, candidates, gold)
                lis.append(LI_HTML.format(data=li_data, context_id=context.id))
                page_cids.append([self.candidates.index(c) for c in candidates])

            # Assemble the page...
            pages.append(PAGE_HTML.format(pid=pid, data=''.join(lis)))
            cids.append(page_cids)
            pid += 1

        # Render in primary Viewer template
        self.cids = cids
        self.html = open(HOME+'/viewer/viewer.html').read() % (self.height, ''.join(pages))
        display(Javascript(open(HOME + '/viewer/viewer.js').read()))

    def _get_labels(self):
        """
        De-serialize labels from Javascript widget, map to internal candidate id, and return as list of tuples
        """
        LABEL_MAP = {'true':1, 'false':-1}
        labels    = [x.split('~~') for x in self._labels_serialized.split(',') if len(x) > 0]
        vals      = [(int(cid), LABEL_MAP.get(l, 0)) for cid,l in labels]
        return vals

    def handle_label_event(self, _, content, buffers):
        """
        Handles label event by persisting new label
        """
        if content.get('event', '') == 'set_label':
            cid = content.get('cid', None)
            value = content.get('value', None)
            if value is True:
                value = 1
            elif value is False:
                value = -1
            else:
                raise ValueError('Unexpected label returned from widget: ' + str(value) +
                                 '. Expected values are True and False.')

            if self.annotations[cid] is not None:
                if self.annotations[cid].value != value:
                    self.annotations[cid].value = value
                    self.session.commit()
            else:
                self.annotations[cid] = Label(key=self.annotator, candidate=self.candidates[cid], value=value)
                self.session.add(self.annotations[cid])
                self.session.commit()
        elif content.get('event', '') == 'delete_label':
            cid = content.get('cid', None)
            self.session.delete(self.annotations[cid])
            self.annotations[cid] = None
            self.session.commit()

    def get_selected(self):
        return self.candidates[self._selected_cid]


class SentenceNgramViewer(Viewer):
    """Viewer for Sentence objects and candidate Spans within them"""
    def __init__(self, candidates, session, gold=[], n_per_page=3, height=225, annotator_name=None):
        super(SentenceNgramViewer, self).__init__(candidates, session, gold=gold, n_per_page=n_per_page, height=height, annotator_name=annotator_name)

    def _is_subspan(self, s, e, c):
        return s >= c.char_start and e <= c.char_end

    def _tag_context(self, sentence, candidates, gold):
        """Tag **potentially overlapping** spans of text, at the character-level"""
        s = sentence.text

        # First, split the sentence into the *smallest* single-candidate chunks
        try:
            both = [c[0] for c in candidates] + [c[1] for c in candidates] \
                        + [g[0] for g in gold] + [g[1] for g in gold]
        except:
            both = [c[0] for c in candidates] + [g[0] for g in gold]
        splits = sorted(list(set([b.char_start for b in both] + [b.char_end + 1 for b in both] + [0, len(s)])))

        # For each chunk, add cid if subset of candidate span, tag if gold, and produce span
        html = ""
        for i in range(len(splits)-1):
            start  = splits[i]
            end    = splits[i+1] - 1

            # Handle both unary and binary candidates
            try:
                # For binary candidates, add classes for both the candidate ID and unary span identifiers
                cids0  = [self.candidates.index(c) for c in candidates if self._is_subspan(start, end, c[0])]
                cids0 += ['%s-0' % cid for cid in cids0]
                cids1  = [self.candidates.index(c) for c in candidates if self._is_subspan(start, end, c[1])]
                cids1 += ['%s-1' % cid for cid in cids1]
                cids   = cids0 + cids1

                # Handle gold...
                gcids = [self.gold.index(g) for g in gold if
                         self._is_subspan(start, end, g[0]) or self._is_subspan(start, end, g[1])]
            except:
                cids  = [self.candidates.index(c) for c in candidates if self._is_subspan(start, end, c[0])]
                gcids = [self.gold.index(g) for g in gold if self._is_subspan(start, end, g[0])]

            html += self._tag_span(s[start:end+1], cids, gold=len(gcids) > 0)
        return html

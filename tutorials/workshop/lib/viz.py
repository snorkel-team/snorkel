from IPython.core.display import display, HTML
from snorkel.lf_helpers import *


def candidate_html(c, label=0, full_sent=True, use_colors=True):
    colors = {1: u"#00e600", 0: u"#CCCCCC", -1: u'#ff4000'}
    div_tmpl = u'''<div style="border: 1px dotted #858585; border-radius:8px;
    background-color:#FDFDFD; padding:5pt 10pt 5pt 10pt">{}</div>'''

    sent_tmpl = u'<p style="font-size:12pt;">{}</p>'
    arg_tmpl = u'<b style="background-color:{};padding:1pt 5pt 1pt 5pt; border-radius:8px">{}</b>'
    chunks = get_text_splits(c)

    text = u""
    for s in chunks[0:]:
        if s in [u"{{A}}", u"{{B}}"]:
            span = c[0].get_span() if s == u"{{A}}" else c[1].get_span()
            text += arg_tmpl.format(colors[label], span)
        else:
            text += s.replace(u"\n", u"<BR/>")
    html = div_tmpl.format(sent_tmpl.format(text.strip()))
    return HTML(html)

def display_candidate(c,label=0):
    display(candidate_html(c,label))
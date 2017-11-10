import csv

import numpy as np

from snorkel.contrib.babble import Explanation
from snorkel.contrib.babble.utils import link_explanation_candidates            
from mturk_processing import MTurkHelper

class GradTurkHelper(MTurkHelper):
    def write_candidate_index(self, fpath):
        with open(fpath, 'wb') as csv_file:
            csvwriter = csv.writer(csv_file)
            for i, c in enumerate(self.candidates):
                csvwriter.writerow([i, c.get_stable_id()])

    def write_candidate_html(self, fpath):
        def highlighted(text):
            """Add yellow highlighting behind the text."""
            return '<span style="background-color: rgb(255, 255, 0);">' + text + '</span>'
        
        def bolded(text):
            return '<b>{}</b>'.format(text)

        def sentenced(candidate):
            """Pull the sentence out of a candidate and highlight its spans."""
            content = candidate.get_parent().text.strip()
            for span in sorted(candidate.get_contexts(), key=lambda x: x.char_start, reverse=True):
                content = content[:span.char_start] + highlighted(span.get_span()) + content[span.char_end + 1:]
            return content.encode('utf-8')

        def paragraphed(text):
            return "<p>{}</p>".format(text)

        paragraphs = []
        for i, c in enumerate(self.candidates):
            paragraphs.append(paragraphed('{}<br/>{}<br/><br/>'.format(bolded(i), sentenced(c))))
        
        body = '\n'.join(paragraphs)

        webpage = ("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Babble Labble Candidates</title>
            <meta charset="UTF-8" />
            
            <style type="text/css">
                * { font-family: Verdana, Arial, sans-serif; }
                body { background-color: #fff; cursor: default; }
                h1 { font-size: 15pt; }
                p { font-size: 10pt; }
            </style>
        </head>

        <body>
            <h1>Babble Labble Candidates</h1>""" + body +
        """
        </body>
        </html>
        """)

        with open(fpath, 'wb') as html_file:
            html_file.write(webpage)

class GradTurkPoster(object):

    def postprocess(self, response_path, candidate_index_path, output_path, candidates):
        with open(candidate_index_path, 'rb') as csv_file:
            csvreader = csv.reader(csv_file)
            candidate_index = [stable_id for i, stable_id in csvreader]

        with open(response_path, 'rb') as csv_file:
            csvreader = UnicodeReader(csv_file)
            csvreader.next()
            label_times = []
            exp_times = []
            exp_all = []
            for (_, name, snorked, start, 
                i1, l1, e1,
                i2, l2, e2,
                i3, l3, e3,
                i4, l4, e4,
                i5, l5, e5,
                end, comments) in csvreader:
                if snorked == 'Yes': 
                    print("Skipping snorkel user {}.".format(name))
                    continue
                start_min = start.split(':')[1]
                end_min = end.split(':')[1]
                idxs = [i1, i2, i3, i4, i5]
                labels = [l1, l2, l3, l4, l5]
                explanations = [e1, e2, e3, e4, e5]
                if not any(explanations):
                    label_times.append(max(1, float(end_min) - float(start_min)))
                else:
                    num_explanations = sum([e != '' for e in explanations])
                    if num_explanations == 5:
                        exp_times.append(max(1, float(end_min) - float(start_min)))
                    for i, exp in enumerate(explanations):
                        condition = unicode(exp)
                        if condition.startswith("I chose this label because "):
                            condition = condition[len("I chose this label because "):]
                        candidate = candidate_index[int(idxs[i])]
                        if labels[i].startswith('True'):
                            label = True
                        elif labels[i].startswith('False'):
                            label = False
                        else:
                            label = None
                        if condition and candidate and label is not None:
                            # import pdb; pdb.set_trace()
                            exp_all.append(Explanation(condition, label, candidate))
            # NOTE: consider whether you want mean or median
            label_average = np.mean(np.array(label_times))/5
            exp_average = np.mean(np.array(exp_times))/5
            
        explanations = link_explanation_candidates(exp_all, candidates)
        return explanations

import csv, codecs, cStringIO

class UTF8Recoder:
    """
    Iterator that reads an encoded stream and reencodes the input to UTF-8
    """
    def __init__(self, f, encoding):
        self.reader = codecs.getreader(encoding)(f)

    def __iter__(self):
        return self

    def next(self):
        return self.reader.next().encode("utf-8")

class UnicodeReader:
    """
    A CSV reader which will iterate over lines in the CSV file "f",
    which is encoded in the given encoding.
    """

    def __init__(self, f, dialect=csv.excel, encoding="utf-8", **kwds):
        f = UTF8Recoder(f, encoding)
        self.reader = csv.reader(f, dialect=dialect, **kwds)

    def next(self):
        row = self.reader.next()
        return [unicode(s, "utf-8") for s in row]

    def __iter__(self):
        return self
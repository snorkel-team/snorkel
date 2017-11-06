import csv

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

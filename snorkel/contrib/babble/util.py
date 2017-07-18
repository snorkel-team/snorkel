import os
import sys
from collections import defaultdict
from subprocess import Popen
import atexit
import requests
import warnings
import json

class CoreNLPHandler(object):
    def __init__(self, tok_whitespace=False, split_newline=False, parse_tree=False):
        # http://stanfordnlp.github.io/CoreNLP/corenlp-server.html
        # Spawn a StanfordCoreNLPServer process that accepts parsing requests at an HTTP port.
        # Kill it when python exits.
        # This makes sure that we load the models only once.
        # In addition, it appears that StanfordCoreNLPServer loads only required models on demand.
        # So it doesn't load e.g. coref models and the total (on-demand) initialization takes only 7 sec.
        self.port = 12345
        self.tok_whitespace = tok_whitespace
        self.split_newline = split_newline
        self.parse_tree = parse_tree
        loc = os.path.join(os.environ['SNORKELHOME'], 'parser')
        cmd = ['java -Xmx4g -cp "%s/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer --port %d --timeout %d > /dev/null'
               % (loc, self.port, 600000)]
        self.server_pid = Popen(cmd, shell=True).pid
        atexit.register(self._kill_pserver)
        props = ''
        if self.tok_whitespace:
            props += '"tokenize.whitespace": "true", '
        if self.split_newline:
            props += '"ssplit.eolonly": "true", '
        annotators = '"tokenize,ssplit,pos,lemma,ner{0}"'.format(
            ',parse' if self.parse_tree else ''
        )
        self.endpoint = 'http://127.0.0.1:%d/?properties={%s"annotators": %s, "outputFormat": "json"}' % (
            self.port, props, annotators
        )

        # Following enables retries to cope with CoreNLP server boot-up latency
        # See: http://stackoverflow.com/a/35504626
        from requests.packages.urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        self.requests_session = requests.Session()
        retries = Retry(total=None,
                        connect=20,
                        read=0,
                        backoff_factor=0.1,
                        status_forcelist=[ 500, 502, 503, 504 ])
        self.requests_session.mount('http://', HTTPAdapter(max_retries=retries))

    def _kill_pserver(self):
        if self.server_pid is not None:
            try:
                os.kill(self.server_pid, signal.SIGTERM)
            except:
                sys.stderr.write('Could not kill CoreNLP server. Might already got killt...\n')

    def parse(self, text):
        """Parse a raw document as a string into a list of sentences"""
        if len(text.strip()) == 0:
            return
        if isinstance(text, unicode):
            text = text.encode('utf-8', 'error')
        resp = self.requests_session.post(self.endpoint, data=text, allow_redirects=True)
        text = text.decode('utf-8')
        content = resp.content.strip()
        if content.startswith("Request is too long"):
            warnings.warn("File {} too long. Max character count is 100K.".format(document.name), RuntimeWarning)
            return
        if content.startswith("CoreNLP request timed out"):
            warnings.warn("CoreNLP request timed out on file {}.".format(document.name), RuntimeWarning)
            return
        try:
            blocks = json.loads(content, strict=False)['sentences']
        except:
            warnings.warn("CoreNLP skipped a malformed sentence.", RuntimeWarning)
            return
        return blocks[0]['tokens']
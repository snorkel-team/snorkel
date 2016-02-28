import atexit
import os
import requests
import signal
import time
from collections import namedtuple, defaultdict
from subprocess import Popen


Sentence = namedtuple('Sentence', 'words, lemmas, poses, dep_parents, dep_labels')


class SentenceParser:
    def __init__(self):
        # http://stanfordnlp.github.io/CoreNLP/corenlp-server.html
        # Spawn a StanfordCoreNLPServer process that accepts parsing requests at an HTTP port.
        # Kill it when python exits.
        # This makes sure that we load the models only once.
        # In addition, it appears that StanfordCoreNLPServer loads only required models on demand.
        # So it doesn't load e.g. coref models and the total (on-demand) initialization takes only 7 sec.
        self.port = 12345
        cmd = ['java -Xmx4g -cp "parser/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer --port %d' % self.port]
        self.server_pid = Popen(cmd, shell=True).pid
        # Wait a bit for java to start up.
        time.sleep(0.5)
        atexit.register(self._kill_pserver)
        self.endpoint = 'http://localhost:%d/?properties={"annotators": "tokenize,ssplit,pos,lemma,depparse", "outputFormat": "conll"}' % self.port

    def _kill_pserver(self):
        if self.server_pid is not None:
            os.kill(self.server_pid, signal.SIGTERM)

    def parse(self, doc):
        """Parse a raw document as a string into a list of sentences"""
        resp = requests.post(self.endpoint, data=doc, allow_redirects=True)
        blocks = resp.content.strip().split('\n\n')
        for block in blocks:
            lines = block.split('\n')
            parts = defaultdict(list)
            for line in lines:
                vals = line.split('\t')
                for i, key in enumerate(['', 'words', 'lemmas', 'poses', '', 'dep_parents', 'dep_labels']):
                    if not key:
                        continue
                    val = vals[i]
                    if key == 'dep_parents':
                        val = int(val)
                    parts[key].append(val)
            sent = Sentence(**parts)
            yield sent


def main():
    doc = 'Hello world. How are you?'
    parser = SentenceParser()
    for s in parser.parse(doc):
        print s


if __name__ == '__main__':
    main()

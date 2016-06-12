# -*- coding: utf-8 -*-

import atexit
import glob
import json
import os
import re
import requests
import signal
import sys
import warnings
from bs4 import BeautifulSoup
from collections import namedtuple, defaultdict
from subprocess import Popen


Sentence = namedtuple('Sentence', ['words', 'lemmas', 'poses', 'dep_parents',
                                   'dep_labels', 'sent_id', 'doc_id', 'text',
                                   'token_idxs', 'doc_name'])

class SentenceParser:
    def __init__(self, tok_whitespace=False):
        # http://stanfordnlp.github.io/CoreNLP/corenlp-server.html
        # Spawn a StanfordCoreNLPServer process that accepts parsing requests at an HTTP port.
        # Kill it when python exits.
        # This makes sure that we load the models only once.
        # In addition, it appears that StanfordCoreNLPServer loads only required models on demand.
        # So it doesn't load e.g. coref models and the total (on-demand) initialization takes only 7 sec.
        self.port = 12345
	self.tok_whitespace = tok_whitespace
        loc = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'parser')
        cmd = ['java -Xmx4g -cp "%s/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer --port %d > /dev/null' % (loc, self.port)]
        self.server_pid = Popen(cmd, shell=True).pid
        atexit.register(self._kill_pserver)
        props = "\"tokenize.whitespace\": \"true\"," if self.tok_whitespace else "" 
        self.endpoint = 'http://127.0.0.1:%d/?properties={%s"annotators": "tokenize,ssplit,pos,lemma,depparse", "outputFormat": "json"}' % (self.port, props)

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

    def parse(self, doc, doc_id=None, doc_name=None):
        """Parse a raw document as a string into a list of sentences"""
        if len(doc.strip()) == 0:
            return
        if isinstance(doc, unicode):
          doc = doc.encode('utf-8')
        resp = self.requests_session.post(self.endpoint, data=doc, allow_redirects=True)
        doc = doc.decode('utf-8')
        content = resp.content.strip()
        if content.startswith("Request is too long") or content.startswith("CoreNLP request timed out"):
          raise ValueError("File {} too long. Max character count is 100K".format(doc_id))
        blocks = json.loads(content, strict=False)['sentences']
        sent_id = 0
        for block in blocks:
            parts = defaultdict(list)
            dep_order, dep_par, dep_lab = [], [], []
            for tok, deps in zip(block['tokens'], block['basic-dependencies']):
                parts['words'].append(tok['word'])
                parts['lemmas'].append(tok['lemma'])
                parts['poses'].append(tok['pos'])
                parts['token_idxs'].append(tok['characterOffsetBegin'])
                dep_par.append(deps['governor'])
                dep_lab.append(deps['dep'])
                dep_order.append(deps['dependent'])
            parts['dep_parents'] = sort_X_on_Y(dep_par, dep_order)
            parts['dep_labels'] = sort_X_on_Y(dep_lab, dep_order)
            parts['sent_id'] = sent_id
            parts['doc_id'] = doc_id
            parts['text'] = doc[block['tokens'][0]['characterOffsetBegin'] : 
                                block['tokens'][-1]['characterOffsetEnd']]
            parts['doc_name'] = doc_name
            sent = Sentence(**parts)
            sent_id += 1
            yield sent
            
'''
Abstract base class for file type parsers
Must implement method inidicating if file can be parsed and parser
'''
class FileTypeReader(object):
    def can_read(self, f):
        raise NotImplementedError()
    def read(self, f):
        raise NotImplementedError()
    def _strip_special(self, s):
        return (''.join(c for c in s if ord(c) < 128)).encode('ascii','ignore')
        
'''
Basic HTML parser using BeautifulSoup to return visible text
'''
class HTMLReader(FileTypeReader):
    def can_read(self, fp):
        return fp.endswith('.html')
    def read(self, fp):
        with open(fp, 'rb') as f:
            mulligatawny = BeautifulSoup(f, 'lxml')
        txt = filter(self._cleaner, mulligatawny.findAll(text=True))
        return ' '.join(self._strip_special(s) for s in txt if s != '\n')
    def _cleaner(self, s):
        if s.parent.name in ['style', 'script', '[document]', 'head', 'title']:
            return False
        elif re.match('<!--.*-->', unicode(s)):
            return False
        return True
        
'''
Text parser for preprocessed files
'''
class TextReader(FileTypeReader):
    def can_read(self, fp):
        return True
    def read(self, fp):
        with open(fp, 'rb') as f:
            return f.read()

'''
Wrapper for a FileTypeParser that parses a file, directory, or pattern
Defaults to using TextParser
'''
class DocParser: 
    def __init__(self, path, ftreader = TextReader()):
        self.path = path
        if not issubclass(ftreader.__class__, FileTypeReader):
            warnings.warn("File parser is not a subclass of FileTypeParser")
        self._ftreader = ftreader
        self._fs = self._get_files()
        
    # Parse all docs parseable by passed file type parser
    def readDocs(self):
        for f in self._fs:
            doc_name = os.path.basename(f)
            if self._ftreader.can_read(f):
                yield doc_name, self._ftreader.read(f)
            else:
                warnings.warn("Skipping unreadable file {}".format(f))
    
    # Use SentenceParser to return parsed sentences
    def parseDocSentences(self):
        sp = SentenceParser()
        return [sent for doc_name, txt in self.readDocs()
                for sent in sp.parse(txt, doc_name, doc_name)]
    
    def _get_files(self):
        if os.path.isfile(self.path):
            return [self.path]
        elif os.path.isdir(self.path):
            return [os.path.join(self.path, f) for f in os.listdir(self.path)]
        else:
            return glob.glob(self.path)
            
    def __repr__(self):
        return "Document parser for files: {}".format(self._fs)

def sort_X_on_Y(X, Y):
  return [x for (y,x) in sorted(zip(Y,X), key=lambda t : t[0])]   

def corenlp_cleaner(words):
  d = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}', '-LCB-': '{',
       '-RSB-': ']', '-LSB-': '['}
  return map(lambda w: d[w] if w in d else w, words)

def main():
    doc = 'Hello world. How are you?'
    parser = SentenceParser()
    for s in parser.parse(doc):
        print s
        
    doc2 = u'IC50 value 87.81 µg mL(-1). IC50 abcµgmue of 1. I µµggval abcµgm ()'
    for s in parser.parse(doc2):
        print s
        print s.text

if __name__ == '__main__':
    main()

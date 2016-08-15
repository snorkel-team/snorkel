# -*- coding: utf-8 -*-

from .models import Corpus, Document, Sentence
import atexit
import warnings
from bs4 import BeautifulSoup
from collections import defaultdict
import glob
import json
import lxml.etree as et
import os
import re
import codecs
import requests
import signal
from subprocess import Popen
import sys


class CorpusParser:
    """
    Invokes a DocParser and runs the output through a SentenceParser to produce a Corpus.
    """

    def __init__(self, doc_parser, sent_parser, max_docs=None):
        self.doc_parser = doc_parser
        self.sent_parser = sent_parser
        self.max_docs = max_docs

    def parse_corpus(self, name=None):
        corpus = Corpus()

        for i, (doc, text) in enumerate(self.doc_parser.parse()):
            if self.max_docs and i == self.max_docs:
                break
            doc.corpus = corpus

            for _ in self.sent_parser.parse(doc, text):
                pass

        if name is not None:
            corpus.name = name

        return corpus


class DocParser:
    """Parse a file or directory of files into a set of Document objects."""
    def __init__(self, path, encoding="utf-8"):
        self.path = path
        self.encoding = encoding
    
    def parse(self):
        """
        Parse a file or directory of files into a set of Document objects.

        - Input: A file or directory path.
        - Output: A set of Document objects, which at least have a _text_ attribute,
                  and possibly a dictionary of other attributes.
        """
        for fp in self._get_files():
            file_name = os.path.basename(fp)
            if self._can_read(file_name):
                for doc, text in self.parse_file(fp, file_name):
                    yield doc, text

    def parse_file(self, fp, file_name):
        raise NotImplementedError()

    def _can_read(self, fpath):
        return True

    def _get_files(self):
        if os.path.isfile(self.path):
            fpaths = [self.path]
        elif os.path.isdir(self.path):
            fpaths = [os.path.join(self.path, f) for f in os.listdir(self.path)]
        else:
            fpaths = glob.glob(self.path)
        if len(fpaths) > 0:
            return fpaths
        else:
            raise IOError("File or directory not found: %s" % (self.path,))


class TextDocParser(DocParser):
    """Simple parsing of raw text files, assuming one document per file"""
    def parse_file(self, fp, file_name):
        with codecs.open(fp, 'rb', self.encoding, errors="ignore") as f:
            name = re.sub(r'\..*$', '', os.path.basename(fp))
            yield Document(name=name, file=file_name, attribs={}), f.read()


class HTMLDocParser(DocParser):
    """Simple parsing of raw HTML files, assuming one document per file"""
    def parse_file(self, fp, file_name):
        with open(fp, 'rb') as f:
            html = BeautifulSoup(f, 'lxml')
            txt = filter(self._cleaner, html.findAll(text=True))
            txt = ' '.join(self._strip_special(s) for s in txt if s != '\n')
            name = re.sub(r'\..*$', '', os.path.basename(fp))
            yield Document(name=name, file=file_name, attribs={}), txt

    def _can_read(self, fpath):
        return fpath.endswith('.html')

    def _cleaner(self, s):
        if s.parent.name in ['style', 'script', '[document]', 'head', 'title']:
            return False
        elif re.match('<!--.*-->', unicode(s)):
            return False
        return True

    def _strip_special(self, s):
        return (''.join(c for c in s if ord(c) < 128)).encode('ascii','ignore')


class XMLDocParser(DocParser):
    """
    Parse an XML file or directory of XML files into a set of Document objects.

    Use XPath queries to specify a _document_ object, and then for each document,
    a set of _text_ sections and an _id_.

    **Note: Include the full document XML etree in the attribs dict with keep_xml_tree=True**
    """
    def __init__(self, path, doc='.//document', text='./text/text()', id='./id/text()',
                    keep_xml_tree=False):
        DocParser.__init__(self, path)
        self.doc = doc
        self.text = text
        self.id = id
        self.keep_xml_tree = keep_xml_tree

    def parse_file(self, f, file_name):
        for i,doc in enumerate(et.parse(f).xpath(self.doc)):
            text = '\n'.join(filter(lambda t : t is not None, doc.xpath(self.text)))
            ids = doc.xpath(self.id)
            id = ids[0] if len(ids) > 0 else None
            # We store the XML tree as a string due to a serialization bug. It cannot currently be pickled directly
            #TODO: Implement a special dictionary that can handle this automatically (http://docs.sqlalchemy.org/en/latest/orm/extensions/mutable.html)
            attribs = {'root': et.tostring(doc)} if self.keep_xml_tree else {}
            yield Document(name=str(id), file=str(file_name), attribs=attribs), str(text)

    def _can_read(self, fpath):
        return fpath.endswith('.xml')


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
        loc = os.path.join(os.environ['SNORKELHOME'], 'parser')
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

    def parse(self, document, text):
        """Parse a raw document as a string into a list of sentences"""
        if len(text.strip()) == 0:
            return
        if isinstance(text, unicode):
          text = text.encode('utf-8')
        resp = self.requests_session.post(self.endpoint, data=text, allow_redirects=True)
        text = text.decode('utf-8')
        content = resp.content.strip()
        if content.startswith("Request is too long") or content.startswith("CoreNLP request timed out"):
          raise ValueError("File {} too long. Max character count is 100K".format(document.id))
        blocks = json.loads(content, strict=False)['sentences']
        sent_id = 0
        diverged = False
        for block in blocks:
            parts = defaultdict(list)
            dep_order, dep_par, dep_lab = [], [], []
            for tok, deps in zip(block['tokens'], block['basic-dependencies']):
                parts['words'].append(tok['word'])
                parts['lemmas'].append(tok['lemma'])
                parts['poses'].append(tok['pos'])
                parts['char_offsets'].append(tok['characterOffsetBegin'])
                dep_par.append(deps['governor'])
                dep_lab.append(deps['dep'])
                dep_order.append(deps['dependent'])
            # make char_offsets relative to start of sentence
            parts['char_offsets'] = [p - parts['char_offsets'][0] for p in parts['char_offsets']]
            parts['dep_parents'] = sort_X_on_Y(dep_par, dep_order)
            parts['dep_labels'] = sort_X_on_Y(dep_lab, dep_order)
            # NOTE: We have observed weird bugs where CoreNLP diverges from raw document text (see Issue #368)
            # In these cases we go with CoreNLP so as not to cause downstream issues but throw a warning
            doc_text = text[block['tokens'][0]['characterOffsetBegin'] : block['tokens'][-1]['characterOffsetEnd']]
            L = len(block['tokens'])
            parts['text'] = ''.join(t['originalText'] + t['after'] if i < L - 1 else t['originalText'] for i,t in enumerate(block['tokens']))
            if not diverged and doc_text != parts['text']:
                diverged = True
                warnings.warn("CoreNLP parse has diverged from raw document text!")
            parts['position'] = sent_id
            sent = Sentence(**parts)
            sent.document = document
            sent_id += 1
            yield sent


def sort_X_on_Y(X, Y):
    return [x for (y,x) in sorted(zip(Y,X), key=lambda t : t[0])]


def corenlp_cleaner(words):
    d = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}', '-LCB-': '{',
         '-RSB-': ']', '-LSB-': '['}
    return map(lambda w: d[w] if w in d else w, words)

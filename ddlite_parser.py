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
import numpy as np
import re
from bs4 import BeautifulSoup
from collections import namedtuple, defaultdict
from subprocess import Popen
import lxml.etree as et
from itertools import chain

id_attrs        = ['id', 'doc_id', 'doc_name']
sentence_attrs  = id_attrs + ['sent_id', 'words', 'lemmas', 'poses', 'dep_parents', 'dep_labels', 'char_offsets', 'text']
table_attrs     = id_attrs + ['table_id', 'cells', 'xhtml'] # 'title', 'caption'
cell_attrs      = sentence_attrs + ['table_id', 'cell_id', 'row_num', 'col_num', \
                  'html_tag', 'html_attrs', 'html_anc_tags', 'html_anc_attrs']

Document = namedtuple('Document', ['id', 'file', 'text', 'attribs'])
Sentence = namedtuple('Sentence', sentence_attrs)
Table    = namedtuple('Table', table_attrs)
Cell     = namedtuple('Cell', cell_attrs)

class DocParser:
    """Parse a file or directory of files into a set of Document objects."""
    def __init__(self, path):
        self.path = path

    def parse(self):
        """
        Parse a file or directory of files into a set of Document objects.

        - Input: A file or directory path.
        - Output: A set of Document objects, which at least have a _text_ attribute,
                  and possibly a dictionary of other attributes.
        """
        for fp in self._get_files():
            file_name = os.path.basename(fp)
            for doc in self.parse_file(fp, file_name):
                yield doc

    def parse_file(self, fp, file_name):
        raise NotImplementedError()

    def _get_files(self):
        if os.path.isfile(self.path):
            return [self.path]
        elif os.path.isdir(self.path):
            return [os.path.join(self.path, f) for f in os.listdir(self.path)]
        else:
            return glob.glob(self.path)

class TextDocParser(DocParser):
    """Simple parsing of raw text files, assuming one document per file"""
    def parse_file(self, fp, file_name):
        with open(fp, 'rb') as f:
            yield Document(id=None, file=file_name, text=f.read(), attribs={})

class HTMLDocParser(DocParser):
    """Simple parsing of raw HTML files, assuming one document per file"""
    def parse_file(self, fp, file_name):
        with open(fp, 'rb') as f:
            soup = BeautifulSoup(f, 'lxml')
            txt = filter(self, soup.findAll(text=True))
            txt = ' '.join(self._strip_special(s) for s in txt if s != '\n')
            yield Document(id=None, file=file_name, text=txt, attribs={})

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
        self.path = path
        self.doc = doc
        self.text = text
        self.id = id
        self.keep_xml_tree = keep_xml_tree

    def parse_file(self, f, file_name):
        for i,doc in enumerate(et.parse(f).xpath(self.doc)):
            #try:
            text = '\n'.join(filter(lambda t : t is not None, doc.xpath(self.text)))
            ids = doc.xpath(self.id)
            id = ids[0] if len(ids) > 0 else None
            attribs = {'root':doc} if self.keep_xml_tree else {}
            yield Document(id=id, file=file_name, text=text, attribs=attribs)
            #except:
            #    print "Error parsing document #%s (id=%s) in file %s" % (i,id,file_name)

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

    def parse(self, s, doc_id=None, doc_name=None):
        """Parse a raw document as a string into a list of sentences"""
        if len(s.strip()) == 0:
            return
        if isinstance(s, unicode):
          s = s.encode('utf-8')
        resp = self.requests_session.post(self.endpoint, data=s, allow_redirects=True)
        s = s.decode('utf-8')
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
                parts['char_offsets'].append(tok['characterOffsetBegin'])
                dep_par.append(deps['governor'])
                dep_lab.append(deps['dep'])
                dep_order.append(deps['dependent'])
            parts['dep_parents'] = sort_X_on_Y(dep_par, dep_order)
            parts['dep_labels'] = sort_X_on_Y(dep_lab, dep_order)
            parts['sent_id'] = sent_id
            parts['doc_id'] = doc_id
            parts['text'] = s[block['tokens'][0]['characterOffsetBegin'] :
                                block['tokens'][-1]['characterOffsetEnd']]
            parts['doc_name'] = doc_name
            parts['id'] = "%s-%s" % (parts['doc_id'], parts['sent_id'])
            sent = Sentence(**parts)
            sent_id += 1
            yield sent

    def parse_docs(self, docs):
        """Parse a list of Document objects into a list of pre-processed Sentences."""
        sents = []
        for doc in docs:
            for sent in self.parse(doc.text, doc.id, doc.file):
                sents.append(sent)
        return sents

class HTMLTableParser(DocParser):
    """Simple parsing of raw HTML tables"""

    def parse_file(self, fp, file_name):
        with open(fp, 'r') as f:
            soup = BeautifulSoup(f, 'xml')
            for doc_id, text in enumerate(soup.find_all('html')):
                yield Document(id=doc_id, file=file_name, text=unicode(text), attribs=None)

class TableParser(SentenceParser):

    def parse_table(self, table, table_id=None, doc_id=None, doc_name=None):
        cells = []
        cell_id = 0
        for row_num, row in enumerate(table.find_all('tr')):
            ancestors = [(row.name, row.attrs.items())] + [(ancestor.name, ancestor.attrs.items()) for ancestor in row.parents if ancestor is not None][:-2]
            (tags, attrs) = zip(*ancestors)
            html_anc_tags = tags
            html_anc_attrs = [a[0]+"="+a[1] for a in chain.from_iterable(attrs)]
            for col_num, cell in enumerate(row.children):
                if cell.name in ['th','td']:
                    cell['cell_id'] = cell_id
                    for sent in self.parse(cell.get_text(strip=True), doc_id, doc_name):
                        parts = sent._asdict()
                        parts['id'] = "%s-%s-%s" % (doc_id, table_id, cell_id)
                        parts['doc_id'] = doc_id
                        parts['table_id'] = table_id
                        parts['sent_id'] = table_id # TEMPORARY! Refactor to just have context_id
                        parts['cell_id'] = cell_id
                        parts['doc_name'] = doc_name
                        parts['row_num'] = row_num
                        parts['col_num'] = col_num
                        parts['html_tag'] = cell.name
                        parts['html_attrs'] = [a[0]+"="+a[1] for a in cell.attrs]
                        parts['html_anc_tags'] = html_anc_tags
                        parts['html_anc_attrs'] = html_anc_attrs
                        cells.append(Cell(**parts))
                    cell_id += 1
        id = "%s-%s" % (doc_id, table_id)
        return Table(id, doc_id, doc_name, table_id, cells, table)

    def parse_docs(self, docs):
        """Parse a list of Document objects into a list of pre-processed Tables."""
        tables = []
        for doc in docs:
            soup = BeautifulSoup(doc.text, 'xml')
            for table_id, table in enumerate(soup.find_all('table')):
                tables.append(self.parse_table(table, table_id=table_id, doc_id=doc.id, doc_name=doc.file))
        return tables

def sort_X_on_Y(X, Y):
    return [x for (y,x) in sorted(zip(Y,X), key=lambda t : t[0])]

def corenlp_cleaner(words):
  d = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}', '-LCB-': '{',
       '-RSB-': ']', '-LSB-': '['}
  return map(lambda w: d[w] if w in d else w, words)


class Corpus(object):
    """
    A Corpus object helps instantiate and then holds a set of Documents and associated Sentences
    Default iterator is over (Document, Sentences) tuples
    """
    def __init__(self, doc_parser, context_parser):

        # Parse documents
        print "Parsing documents..."
        self._docs_by_id = {}
        for doc in doc_parser.parse():
            self._docs_by_id[doc.id] = doc

        # Parse contexts
        print "Parsing contexts..."
        self._sentences_by_id     = {}
        self._sentences_by_doc_id = defaultdict(list)
        for context in context_parser.parse_docs(self._docs_by_id.values()):
            self._sentences_by_id[context.id] = context
            self._sentences_by_doc_id[context.doc_id].append(context)

    def __iter__(self):
        """Default iterator is over (document, sentence) tuples"""
        for doc in self.iter_docs():
            yield (doc, self.get_sentences_in(doc.id))

    def iter_docs(self):
        return self._docs_by_id.itervalues()

    def iter_sentences(self):
        return self._sentences_by_id.itervalues()

    def get_docs(self):
        return self._docs_by_id.values()

    def get_sentences(self):
        return self._sentences_by_id.values()

    def get_doc(self, id):
        return self._docs_by_id[id]

    def get_sentence(self, id):
        return self._sentences_by_id[id]

    def get_sentences_in(self, doc_id):
        return self._sentences_by_doc_id[doc_id]

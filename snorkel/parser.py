# -*- coding: utf-8 -*-

from .models import Corpus, Document, Sentence, Table, Row, Col, Cell, Phrase, construct_stable_id
import atexit
import warnings
from bs4 import BeautifulSoup, NavigableString, Tag
from collections import defaultdict
import glob
import json
import lxml.etree as et
import os
import re
import requests
import signal
import codecs
from subprocess import Popen
import sys
from .utils import ProgressBar, sort_X_on_Y, split_html_attrs
import copy # TODO: remove once OmniParser is fixed



class CorpusParser:
    """Invokes a DocParser and runs the output through a SentenceParser to produce a Corpus."""
    def __init__(self, doc_parser, sent_parser, max_docs=None):
        self.doc_parser = doc_parser
        self.sent_parser = sent_parser
        self.max_docs = max_docs

    def parse_corpus(self, name, session=None):
        corpus = Corpus(name=name)
        if session is not None:
            session.add(corpus)
        if self.max_docs is not None:
            pb = ProgressBar(self.max_docs)
        for i, (doc, text) in enumerate(self.doc_parser.parse()):
            if self.max_docs is not None:
                pb.bar(i)
                if i == self.max_docs:
                    break
            corpus.append(doc)
            for _ in self.sent_parser.parse(doc, text):
                pass
        if self.max_docs is not None:
            pb.close()
        if session is not None:
            session.commit()
        # corpus.stats() # Note: corpus.stats breaks with OmniParser
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

    def get_stable_id(self, doc_id):
        return "%s::document:0:0" % doc_id

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

class TSVDocParser(DocParser):
    """Simple parsing of TSV file with one (doc_name <tab> doc_text) per line"""
    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as tsv:
            for line in tsv:
                (doc_name, doc_text) = line.split('\t')
                stable_id=self.get_stable_id(doc_name)
                yield Document(name=doc_name, stable_id=stable_id, meta={'file_name' : file_name}), doc_text


class TextDocParser(DocParser):
    """Simple parsing of raw text files, assuming one document per file"""
    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as f:
            name      = re.sub(r'\..*$', '', os.path.basename(fp))
            stable_id = self.get_stable_id(name)
            yield Document(name=name, stable_id=stable_id, meta={'file_name' : file_name}), f.read()


class HTMLDocParser(DocParser):
    """Simple parsing of raw HTML files, assuming one document per file"""
    def parse_file(self, fp, file_name):
        with open(fp, 'rb') as f:
            html = BeautifulSoup(f, 'lxml')
            txt = filter(self._cleaner, html.findAll(text=True))
            txt = ' '.join(self._strip_special(s) for s in txt if s != '\n')
            name = re.sub(r'\..*$', '', os.path.basename(fp))
            stable_id = self.get_stable_id(name)
            yield Document(name=name, stable_id=stable_id, meta={'file_name' : file_name}), txt

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


class XMLMultiDocParser(DocParser):
    """
    Parse an XML file _which contains multiple documents_ into a set of Document objects.

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
            doc_id = str(doc.xpath(self.id)[0])
            text   = '\n'.join(filter(lambda t : t is not None, doc.xpath(self.text)))
            meta = {'file_name': str(file_name)}
            if self.keep_xml_tree:
                meta['root'] = et.tostring(doc)
            stable_id = self.get_stable_id(doc_id)
            yield Document(name=doc_id, stable_id=stable_id, meta=meta), text

    def _can_read(self, fpath):
        return fpath.endswith('.xml')


PTB = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}', '-LCB-': '{',
         '-RSB-': ']', '-LSB-': '['}

class CoreNLPHandler:
    def __init__(self, delim='', tok_whitespace=False):
        # http://stanfordnlp.github.io/CoreNLP/corenlp-server.html
        # Spawn a StanfordCoreNLPServer process that accepts parsing requests at an HTTP port.
        # Kill it when python exits.
        # This makes sure that we load the models only once.
        # In addition, it appears that StanfordCoreNLPServer loads only required models on demand.
        # So it doesn't load e.g. coref models and the total (on-demand) initialization takes only 7 sec.
        self.port = 12345
        self.tok_whitespace = tok_whitespace
        loc = os.path.join(os.environ['SNORKELHOME'], 'parser')
        cmd = ['java -Xmx4g -cp "%s/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer --port %d --timeout %d > /dev/null'
               % (loc, self.port, 300000)]
        self.server_pid = Popen(cmd, shell=True).pid
        atexit.register(self._kill_pserver)
        props = "\"tokenize.whitespace\": \"true\"," if self.tok_whitespace else ""
        props += "\"ssplit.htmlBoundariesToDiscard\": \"%s\"," % delim if delim else ""
        self.endpoint = 'http://127.0.0.1:%d/?properties={%s"annotators": "tokenize,ssplit,pos,lemma,depparse,ner", "outputFormat": "json"}' % (self.port, props)

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
            text = text.encode('utf-8', 'error')
        resp = self.requests_session.post(self.endpoint, data=text, allow_redirects=True)
        text = text.decode('utf-8')
        content = resp.content.strip()
        if content.startswith("Request is too long"):
            raise ValueError("File {} too long. Max character count is 100K.".format(document.name))
        if content.startswith("CoreNLP request timed out"):
            raise ValueError("CoreNLP request timed out on file {}.".format(document.name))
        try:
            blocks = json.loads(content, strict=False)['sentences']
        except:
            print "SKIPPED A MALFORMED SENTENCE!"
            return
        position = 0
        diverged = False
        for block in blocks:
            parts = defaultdict(list)
            dep_order, dep_par, dep_lab = [], [], []
            for tok, deps in zip(block['tokens'], block['basic-dependencies']):
                parts['words'].append(tok['word'])
                parts['lemmas'].append(tok['lemma'])
                parts['pos_tags'].append(tok['pos'])
                parts['ner_tags'].append(tok['ner'])
                parts['char_offsets'].append(tok['characterOffsetBegin'])
                dep_par.append(deps['governor'])
                dep_lab.append(deps['dep'])
                dep_order.append(deps['dependent'])

            # make char_offsets relative to start of sentence
            abs_sent_offset = parts['char_offsets'][0]
            parts['char_offsets'] = [p - abs_sent_offset for p in parts['char_offsets']]
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
            parts['position'] = position
            
            # replace PennTreeBank tags with original forms
            parts['words'] = [PTB[w] if w in PTB else w for w in parts['words']]
            parts['lemmas'] = [PTB[w.upper()] if w.upper() in PTB else w for w in parts['lemmas']]

            # Link the sentence to its parent document object
            parts['document'] = document

            # Assign the stable id as document's stable id plus absolute character offset
            abs_sent_offset_end = abs_sent_offset + parts['char_offsets'][-1] + len(parts['words'][-1])
            parts['stable_id'] = construct_stable_id(document, 'sentence', abs_sent_offset, abs_sent_offset_end)
            position += 1
            yield parts


class SentenceParser(object):
    def __init__(self, tok_whitespace=False):
        self.corenlp_handler = CoreNLPHandler(tok_whitespace=tok_whitespace)

    def parse(self, doc, text):
        """Parse a raw document as a string into a list of sentences"""
        for parts in self.corenlp_handler.parse(doc, text):
            yield Sentence(**parts)


class HTMLParser(DocParser):
    """Simple parsing of files into html documents"""
    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as f:
            soup = BeautifulSoup(f, 'lxml')
            for text in soup.find_all('html'):
                name = os.path.basename(fp)[:os.path.basename(fp).rfind('.')]
                stable_id = self.get_stable_id(name)
                yield Document(name=name, stable_id=stable_id, meta={'file_name' : file_name}), unicode(text)


class OmniParser(object):
    def __init__(self):
        self.delim = "<NC>" # NC = New Cell 
        self.corenlp_handler = CoreNLPHandler(delim=self.delim[1:-1])

    def parse(self, document, text):
        soup = BeautifulSoup(text, 'lxml')
        self.table_idx = -1
        self.phrase_idx = 0      
        for phrase in self.parse_tag(soup, document):
            yield phrase

    def parse_tag(self, tag, document, table=None, row=None, col=None, cell=None, anc_tags=[], anc_attrs=[]):
        if any(isinstance(child, NavigableString) and unicode(child)!=u'\n' for child in tag.contents):
            # NOTE: do '?' replacement for hardware only
            text = tag.get_text(' ').replace('?',' - ')
            tag.clear()
            tag.string = text
        for child in tag.contents:
            if isinstance(child, NavigableString):
                for parts in self.corenlp_handler.parse(document, unicode(child)):
                    parts['document'] = document
                    parts['table'] = table
                    parts['cell'] = cell
                    parts['row'] = row
                    parts['col'] = col
                    parts['phrase_id'] = self.phrase_idx
                    parts['row_num'] = row.position if row is not None else None
                    parts['col_num'] = col.position if col is not None else None
                    parts['html_tag'] = tag.name
                    parts['html_attrs'] = tag.attrs
                    parts['html_anc_tags'] = anc_tags
                    parts['html_anc_attrs'] = anc_attrs
                    parts['stable_id'] = "%s::%s:%s:%s" % (document.name, 'phrase', self.phrase_idx, self.phrase_idx)
                    yield Phrase(**parts)
                    self.phrase_idx += 1
            else: # isinstance(child, Tag) = True
                if child.name == "table":
                    self.table_idx += 1
                    self.row_num = -1
                    # self.cell_idx = -1
                    stable_id = "%s::%s:%s:%s" % (document.name, 'table', self.table_idx, self.table_idx)
                    table = Table(document=document, stable_id=stable_id, position=self.table_idx, text=unicode(child))
                elif child.name == "tr":
                    self.row_num += 1
                    stable_id = "%s::%s:%s:%s" % (document.name, 'row', table.position, self.row_num)
                    row = Row(document=document, table=table, stable_id=stable_id, position=self.row_num)
                    self.col_num = -1
                elif child.name in ["td","th"]:
                    # self.cell_idx += 1
                    self.col_num += 1
                    if len(table.cols) <= self.col_num:
                        stable_id = "%s::%s:%s:%s" % (document.name, 'col', table.position, self.col_num)
                        col = Col(document=document, table=table, stable_id=stable_id, position=self.col_num)
                    else:
                        col = table.cols[self.col_num]
                        # TODO: remove after testing
                        assert(col.position==self.col_num)
                    parts = defaultdict(list)
                    parts['document'] = document
                    parts['table'] = table
                    parts['row'] = row
                    parts['col'] = col
                    # parts['position'] = self.cell_idx
                    parts['text'] = unicode(child)
                    parts['html_tag'] = child.name
                    parts['html_attrs'] = split_html_attrs(child.attrs.items())
                    parts['html_anc_tags'] = anc_tags 
                    parts['html_anc_attrs'] = anc_attrs
                    parts['stable_id'] = "%s::%s:%s:%s:%s" % (document.name, 'cell', table.position, row.position, col.position)
                    cell = Cell(**parts)
                # FIXME: making so many copies is hacky and wasteful; use a stack?
                temp_anc_tags = copy.deepcopy(anc_tags)
                temp_anc_tags.append(child.name)
                temp_anc_attrs = copy.deepcopy(anc_attrs)
                temp_anc_attrs.extend(split_html_attrs(child.attrs.items()))
                for phrase in self.parse_tag(child, document, table, row, col, cell, temp_anc_tags, temp_anc_attrs):
                    yield phrase

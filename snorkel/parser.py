# -*- coding: utf-8 -*-

from .models import Corpus, Document, Sentence, Table, Cell, Phrase
import atexit
from bs4 import BeautifulSoup, NavigableString, Tag
from collections import defaultdict
from itertools import chain
import glob
import json
import lxml.etree as et
import os
import re
import requests
import signal
from subprocess import Popen
import lxml.etree as et
from itertools import chain
from utils import corenlp_cleaner, sort_X_on_Y, split_html_attrs
import sys



class CorpusParser:
    """
    Invokes a DocParser and runs the output through a ContextParser
    (e.g., SentenceParser) to produce a Corpus.
    """

    def __init__(self, doc_parser, context_parser, max_docs=None):
        self.doc_parser = doc_parser
        self.context_parser = context_parser
        self.max_docs = max_docs

    def parse_corpus(self, name=None):
        corpus = Corpus()

        for i, (doc, text) in enumerate(self.doc_parser.parse()):
            if self.max_docs and i == self.max_docs:
                break
            doc.corpus = corpus

            for _ in self.context_parser.parse(doc, text):
                pass

        if name is not None:
            corpus.name = name

        return corpus


class DocParser:
    """Parse a file or directory of files into a set of Document objects."""
    def __init__(self, path):
        self.path = path
        self.init()

    def init(self):
        pass

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
        with open(fp, 'rb') as f:
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


class SentenceParser(object):
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
        cmd = ['java -Xmx4g -cp "%s/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer --port %d > /dev/null' % (loc, self.port)]
        self.server_pid = Popen(cmd, shell=True).pid
        atexit.register(self._kill_pserver)
        props = "\"tokenize.whitespace\": \"true\"," if self.tok_whitespace else ""
        props += "\"ssplit.htmlBoundariesToDiscard\": \"%s\"," % delim if delim else ""
        # props += "\"ssplit.newlineIsSentenceBreak\": \"%s\"," % "two" if delim else ""
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

    def get_nlp_tags(self, text, document):
        if len(text.strip()) == 0:
            return
        if isinstance(text, unicode):
            text = text.encode('utf-8')
        resp = self.requests_session.post(self.endpoint, data=text, allow_redirects=True)
        text = text.decode('utf-8')
        content = resp.content.strip()
        if content.startswith("Request is too long") or content.startswith("CoreNLP request timed out"):
          raise ValueError("File {} too long. Max character count is 100K".format(document.name))
        blocks = json.loads(content, strict=False)['sentences']
        position = 0
        for block in blocks:
            parts = defaultdict(list)
            dep_order, dep_par, dep_lab = [], [], []
            num_tokens = len(block['tokens'])
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
            parts['text'] = text[block['tokens'][0]['characterOffsetBegin'] :
                                block['tokens'][-1]['characterOffsetEnd']]
            parts['xmltree'] = None
            parts['position'] = position
            position += 1
            yield parts

    def parse(self, document, text):
        """Parse a raw document as a string into a list of sentences"""
        for parts in self.get_nlp_tags(text, document):
            sent = Sentence(**parts)
            sent.document = document
            yield sent


class HTMLParser(DocParser):
    """Simple parsing of files into html documents"""
    # TEMP
    # def init(self):
    #     self.doc_id = 0
    # TEMP

    def parse_file(self, fp, file_name):
        with open(fp, 'r') as f:
            soup = BeautifulSoup(f, 'lxml')
            for text in soup.find_all('html'):
                # id = self.doc_id
                name = re.sub(r'\..*$', '', os.path.basename(fp))
                attribs = None
                yield Document(name=name, file=str(file_name), attribs=attribs), str(text)
                # self.doc_id += 1


class OmniParser(object):
    def __init__(self):
        self.table_parser = TableParser()

    def parse(self, document, text):
        soup = BeautifulSoup(text, 'lxml')
        for phrase in self.parse_tag(document, soup):
            yield phrase

    def parse_tag(self, document, tag):
        for child in tag.contents:
            if isinstance(child, NavigableString):
                pass
                # print unicode(child)
            elif isinstance(child, Tag):
                if child.name == "table":
                    print "TABLE!"
                    for phrase in self.table_parser.parse(document, unicode(child)):
                        yield phrase
                else:
                    for phrase in self.parse_tag(document, child):
                        yield phrase
            else:
                import pdb; pdb.set_trace()


class TableParser(SentenceParser):
    """Simple parsing of the tables in html documents into cells and phrases within cells"""
    def __init__(self, tok_whitespace=False):
        self.delim = "<NC>" # NC = New Cell
        super(TableParser, self).__init__(delim=self.delim[1:-1], tok_whitespace=tok_whitespace)

    def parse(self, document, text, batch=False):
        # BROKEN: DO NOT USE BATCH MODE FOR NOW
        if batch:
            for table in self.parse_html(document, text):
                char_idx = 0
                cell_start = [char_idx]
                for cell in self.parse_table(table):
                    char_idx += len(cell.text)
                    cell_start.append(char_idx)
                text_batch = self.delim.join(cell.text for cell in table.cells)
                char_idx = 0
                cell_idx = 0
                position = 0 # position of Phrase in Cell
                for parts in self.get_nlp_tags(text_batch, document):
                    while char_idx >= cell_start[cell_idx + 1]:
                        cell_idx += 1
                        position = 0
                    parts['position'] = position
                    char_idx += len(parts['text']) + (position > 0) # account for lost whitespace
                    position += 1
                    yield Phrase(**(self.inherit_cell_attrs(table.cells[cell_idx], parts)))

        else: # not batched
            for table in self.parse_html(document, text):
                for cell in self.parse_table(table):
                    for phrase in self.parse_cell(cell):
                        yield phrase

    def parse_html(self, document, text):
        soup = BeautifulSoup(text, 'lxml')
        for i, table in enumerate(soup.find_all('table')):
            yield Table(document_id=document.id,
                        document=document,
                        position=i,
                        text=str(table))

    def parse_table(self, table):
        soup = BeautifulSoup(table.text, 'lxml')
        position = 0
        for row_num, row in enumerate(soup.find_all('tr')):
            ancestors = ([(row.name, row.attrs.items())]
                + [(ancestor.name, ancestor.attrs.items())
                for ancestor in row.parents if ancestor is not None][:-2])
            (tags, attrs) = zip(*ancestors)
            html_anc_tags = tags
            html_anc_attrs = split_html_attrs(chain.from_iterable(attrs))
            col_num = 0
            for html_cell in row.children:
                # TODO: include title, caption, footers, etc.
                if html_cell.name in ['th','td']:
                    parts = defaultdict(list)
                    parts['document_id'] = table.document_id
                    parts['table_id'] = table.id
                    parts['position'] = position
                    parts['document'] = table.document
                    parts['table'] = table

                    parts['text'] = str(html_cell.get_text(strip=True))
                    parts['row_num'] = row_num
                    parts['col_num'] = col_num
                    parts['html_tag'] = html_cell.name
                    parts['html_attrs'] = split_html_attrs(html_cell.attrs.items())
                    parts['html_anc_tags'] = html_anc_tags
                    parts['html_anc_attrs'] = html_anc_attrs
                    cell = Cell(**parts)
                    html_cell['snorkel_id'] = cell.id   # add new attribute to the html
                    yield cell
                    position += 1
                    col_num += 1

    def parse_cell(self, cell):
        parts = self.inherit_cell_attrs(cell, defaultdict(list))
        for i, sent in enumerate(super(TableParser, self).parse(cell.document, cell.text)):
            parts['text'] = sent.text
            parts['position'] = i
            parts['words'] = sent.words
            parts['lemmas'] = sent.lemmas
            parts['poses'] = sent.poses
            parts['char_offsets'] = sent.char_offsets
            parts['dep_parents'] = sent.dep_parents
            parts['dep_labels'] = sent.dep_labels
            yield Phrase(**parts)

    def inherit_cell_attrs(self, cell, parts):
        parts['document_id'] = cell.document_id
        parts['table_id'] = cell.table_id
        parts['cell_id'] = cell.id

        parts['document'] = cell.document
        parts['table'] = cell.table
        parts['cell'] = cell

        parts['row_num'] = cell.row_num
        parts['col_num'] = cell.col_num
        parts['html_tag'] = cell.html_tag
        parts['html_attrs'] = cell.html_attrs
        parts['html_anc_tags'] = cell.html_anc_tags
        parts['html_anc_attrs'] = cell.html_anc_attrs
        return parts

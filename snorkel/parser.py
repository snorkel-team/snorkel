# -*- coding: utf-8 -*-

from .models import Corpus, Document, Sentence, Table, Cell, Phrase
import atexit
from bs4 import BeautifulSoup
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
import sys
# =======
# import warnings
# import numpy as np
# >>>>>>> tables


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

# =======
# id_attrs        = ['id', 'doc_id', 'doc_name']
# lingual_attrs   = ['words', 'lemmas', 'poses', 'dep_parents', 'dep_labels', 'char_offsets', 'text']
# sentence_attrs  = id_attrs + ['sent_id'] + lingual_attrs
# table_attrs     = id_attrs + ['context_id', 'table_id', 'phrases', 'html']
# cell_attrs      = id_attrs + ['context_id', 'table_id', 'cell_id', 'row_num', 'col_num', \
#                   'html_tag', 'html_attrs', 'html_anc_tags', 'html_anc_attrs']
# phrase_attrs    = cell_attrs + ['phrase_id', 'sent_id'] + lingual_attrs


# Document = namedtuple('Document', ['id', 'file', 'text', 'attribs'])
# Sentence = namedtuple('Sentence', sentence_attrs)
# Table    = namedtuple('Table', table_attrs)
# Cell     = namedtuple('Cell', cell_attrs)
# Phrase   = namedtuple('Phrase', phrase_attrs)
# >>>>>>> tables

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
            parts['text'] = text[block['tokens'][0]['characterOffsetBegin'] :
                                block['tokens'][-1]['characterOffsetEnd']]
            parts['position'] = sent_id
            sent = Sentence(**parts)
            sent.document = document
            sent_id += 1
            yield sent


class HTMLParser(DocParser):
    """Simple parsing of files into html documents"""
    def init(self):
        self.doc_id = 0

    def parse_file(self, fp, file_name):
        with open(fp, 'r') as f:
            soup = BeautifulSoup(f, 'xml') # TODO: consider change from XML to HTML?
            for text in enumerate(soup.find_all('html')):
                id = self.doc_id
                attribs = None
                yield Document(name=str(id), file=str(file_name), attribs=attribs), str(text)
                self.doc_id += 1
                # TODO: need unicode instead of str for messy htmls?


class TableParser(SentenceParser):
    """Simple parsing of the tables in html documents into cells and phrases within cells"""
    def parse(self, document, text):
        for table in self.parse_html(document, text):
            for cell in self.parse_table(table):
                for phrase in self.parse_cell(cell):
                    yield phrase

    def parse_html(self, document, text):
        soup = BeautifulSoup(text, 'lxml') # TODO: lxml is best parser for this?
        for i, table in enumerate(soup.find_all('table')):
            yield Table(document_id=document.id,
                        document=document,
                        position=i,
                        text=str(text))

    def parse_table(self, table):
        soup = BeautifulSoup(table.text, 'lxml') # TODO: lxlml is best parser for this?
        position = 0
        for row_num, row in enumerate(soup.find_all('tr')):
            ancestors = ([(row.name, row.attrs.items())]
                + [(ancestor.name, ancestor.attrs.items())
                for ancestor in row.parents if ancestor is not None][:-2])
            (tags, attrs) = zip(*ancestors)
            html_anc_tags = tags
            html_anc_attrs = []
            for a in chain.from_iterable(attrs):
                attr = a[0]
                values = a[1]
                if isinstance(values, list):
                    html_anc_attrs += ["=".join([attr,val]) for val in values]
                else:
                    html_anc_attrs += ["=".join([attr,values])]
            for col_num, html_cell in enumerate(row.children):
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
                    html_attrs = []
                    # TODO: clean this
                    for a in html_cell.attrs.items():
                        attr = a[0]
                        values = a[1]
                        if isinstance(values, list):
                            html_attrs += ["=".join([attr,val]) for val in values]
                        else:
                            html_attrs += ["=".join([attr,values])]
                    parts['html_attrs'] = html_attrs
                    parts['html_anc_tags'] = html_anc_tags
                    parts['html_anc_attrs'] = html_anc_attrs
                    cell = Cell(**parts)
                    # add new attribute to the html
                    html_cell['snorkel_id'] = cell.id
                    yield cell

    def parse_cell(self, cell):
        parts = defaultdict(list)
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

    # def parse_table(self, table, table_idx=None, doc_id=None, doc_name=None):
    #     table_id = "%s-%s" % (doc_id, table_idx)
    #     phrases = {}
    #     cell_idx = -1
    #     for row_num, row in enumerate(table.find_all('tr')):
    #         ancestors = ([(row.name, row.attrs.items())]
    #             + [(ancestor.name, ancestor.attrs.items())
    #             for ancestor in row.parents if ancestor is not None][:-2])
    #         (tags, attrs) = zip(*ancestors)
    #         html_anc_tags = tags
    #         html_anc_attrs = []
    #         for a in chain.from_iterable(attrs):
    #             attr = a[0]
    #             values = a[1]
    #             if isinstance(values, list):
    #                 html_anc_attrs += ["=".join([attr,val]) for val in values]
    #             else:
    #                 html_anc_attrs += ["=".join([attr,values])]
    #         for col_num, cell in enumerate(row.children):
    #             # NOTE: currently not including title, caption, footers, etc.
    #             cell_idx += 1
    #             if cell.name in ['th','td']:
    #                 cell_id = "%s-%s" % (table_id, cell_idx)
    #                 for phrase_idx, phrase in enumerate(self.parse(cell.get_text(strip=True), doc_id, doc_name)):
    #                     phrase_id = "%s-%s" % (cell_id, phrase_idx)
    #                     parts = phrase._asdict()
    #                     parts['doc_id'] = doc_id
    #                     parts['table_id'] = table_id
    #                     parts['context_id'] = table_id # tables are the context
    #                     parts['cell_id'] = cell_id
    #                     parts['phrase_id'] = phrase_id
    #                     parts['sent_id'] = phrase_id # temporary fix until ORM
    #                     parts['id'] = phrase_id
    #                     parts['doc_name'] = doc_name
    #                     parts['row_num'] = row_num
    #                     parts['col_num'] = col_num
    #                     parts['html_tag'] = cell.name
    #                     html_attrs = []
    #                     for a in cell.attrs.items():
    #                         attr = a[0]
    #                         values = a[1]
    #                         if isinstance(values, list):
    #                             html_attrs += ["=".join([attr,val]) for val in values]
    #                         else:
    #                             html_attrs += ["=".join([attr,values])]
    #                     parts['html_attrs'] = html_attrs
    #                     parts['html_anc_tags'] = html_anc_tags
    #                     parts['html_anc_attrs'] = html_anc_attrs
    #                     phrases[phrase_id] = Phrase(**parts)
    #                 # add new attribute to the html
    #                 cell['cell_id'] = cell_id
    #     context_id = table_id
    #     id = table_id
    #     return Table(id, doc_id, doc_name, context_id, table_id, phrases, str(table))

    # def parse_docs(self, docs):
    #     """Parse a list of Document objects into a list of pre-processed Tables."""
    #     tables = []
    #     for doc in docs:
    #         soup = BeautifulSoup(doc.text, 'lxml')
    #         for table_idx, table in enumerate(soup.find_all('table')):
    #             tables.append(self.parse_table(table, table_idx=table_idx, doc_id=doc.id, doc_name=doc.file))
    #     return tables


class PassParser(TableParser):
    def parse_table(self, table, table_idx=None, doc_id=None, doc_name=None):
        pass


def sort_X_on_Y(X, Y):
    return [x for (y,x) in sorted(zip(Y,X), key=lambda t : t[0])]

def corenlp_cleaner(words):
    d = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}', '-LCB-': '{',
         '-RSB-': ']', '-LSB-': '['}
    return map(lambda w: d[w] if w in d else w, words)

# =======
# class Corpus(object):
#     """
#     A Corpus object helps instantiate and then holds a set of Documents and associated Contexts
#     Default iterator is over (Document, Contexts) tuples
#     """

#     def __init__(self, doc_parser, context_parser, max_docs=None):

#         # Parse documents
#         print "Parsing documents..."
#         self._docs_by_id = {}
#         self.num_docs = 0
#         for i, doc in enumerate(doc_parser.parse()):
#             if max_docs and i == max_docs:
#                 break
#             self._docs_by_id[doc.id] = doc
#             self.num_docs += 1

#         # Parse sentences
#         print "Parsing contexts..."
#         self._contexts_by_id     = {}
#         self._contexts_by_doc_id = defaultdict(list)
#         self.num_contexts = 0
#         for context in context_parser.parse_docs(self._docs_by_id.values()):
#             self._contexts_by_id[context.id] = context
#             self._contexts_by_doc_id[context.doc_id].append(context)
#             self.num_contexts += 1
#         print "Parsed %s documents and %s contexts" % (self.num_docs, self.num_contexts)

#     def __iter__(self):
#         """Default iterator is over (document, context) tuples"""
#         for doc in self.iter_docs():
#             yield (doc, self.get_contexts_in(doc.id))

#     def iter_docs(self):
#         return self._docs_by_id.itervalues()

#     def iter_contexts(self):
#         return self._contexts_by_id.itervalues()

#     def get_docs(self):
#         return self._docs_by_id.values()
# >>>>>>> tables




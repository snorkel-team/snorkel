# -*- coding: utf-8 -*-

from .models import Corpus, Document, Webpage, Sentence, Table, Cell, Phrase, construct_stable_id, split_stable_id
from snorkel.utils import ProgressBar, sort_X_on_Y, split_html_attrs
from .visual import VisualLinker
import atexit
import warnings
from bs4 import BeautifulSoup, NavigableString, Tag, Comment
from lxml.html import fromstring
from lxml import etree
from collections import defaultdict
import itertools
import glob
import json
import lxml.etree as et
import numpy as np
import os
import re
import requests
import signal
import codecs
from subprocess import Popen
import sys
import gzip
import json
from timeit import default_timer as timer
from requests.packages.urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

class CorpusParser:
    """Invokes a DocParser and runs the output through a ContextParser to produce a Corpus."""
    def __init__(self, doc_parser, context_parser, max_docs=None):
        self.doc_parser = doc_parser
        self.context_parser = context_parser
        self.max_docs = max_docs

    def parse_corpus(self, session, name):
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
            for _ in self.context_parser.parse(doc, text):
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

class MemexParser(DocParser):
    """Simple parsing of JSON file with one (doc_name <tab> doc_text) per line"""    
    def parse_file(self, fp, file_name):
        with gzip.open(fp, 'rt') as r:
            for line in r:
                obj = json.loads(line, encoding="utf-8", strict=True)
                # if object does not have _id, we skip it
                if not '_id' in obj: continue
                id = obj['_id']

                # if object does not have _source or _source.url, we skip it
                if not '_source' in obj or not 'url' in obj['_source']: continue

                # if it's not a escorts document, we skip it
                # if obj['_type'] != 'escorts': continue
                type = obj['_type']

                # if we don't have a site module for it, we skip it
                source = obj['_source']
                url = source['url']

                if 'crawl_data' in source and 'status' in source['crawl_data']:
                    status = source['crawl_data']['status']
                    if status == 403: continue

                # if there's no raw content
                if 'raw_content' not in source or source['raw_content'].strip() =='': continue

                raw_content = source['raw_content']
                crawltime = source['timestamp']
                stable_id = self.get_stable_id(id)
                yield (Webpage(name=id, url=url, page_type=type, raw_content=raw_content, 
                            crawltime=crawltime, all=line, stable_id=stable_id), raw_content)


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


PTB = {'-RRB-': u')', '-LRB-': u'(', '-RCB-': u'}', '-LCB-': u'{', '-RSB-': u']',
       '-LSB-': u'['}

class CoreNLPHandler:
    '''
    Connects to an existing instance of CoreNLP server
    '''
    def __init__(self, delim='', tok_whitespace=False):
        # http://stanfordnlp.github.io/CoreNLP/corenlp-server.html
        self.port = int(os.environ.get('JAVANLPPORT',12345))
        self.tok_whitespace = tok_whitespace
        props = "\"tokenize.whitespace\": \"true\"," if self.tok_whitespace else ""
        props += "\"ssplit.htmlBoundariesToDiscard\": \"%s\"," % delim if delim else ""
        self.endpoint = 'http://127.0.0.1:%d/?properties={%s\
                        "annotators": "tokenize,ssplit,pos,lemma,depparse,ner",\
                        "outputFormat": "json"}' % (self.port, props)

        # Following enables retries to cope with CoreNLP server boot-up latency
        # See: http://stackoverflow.com/a/35504626
        self.requests_session = requests.Session()
        retries = Retry(total=None,
                        connect=20,
                        read=0,
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504])
        self.requests_session.mount('http://', HTTPAdapter(max_retries=retries))

        self.ptb_rgx = re.compile(r'-[A-Z]{2}B-')
                
    def parse(self, document, text):
        """Parse a raw document as a string into a list of sentences"""
        def ptb_clean(text):
            for ptb_match in self.ptb_rgx.finditer(text): 
                text = text.replace(ptb_match.group(0), PTB[ptb_match.group(0)])
            return text
        
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
            warnings.warn("CoreNLP skipped a malformed sentence.", RuntimeWarning)
            return
        position = 0
        for block in blocks:
            parts = defaultdict(list)
            dep_order, dep_par, dep_lab = [], [], []
            for tok, deps in zip(block['tokens'], block['basic-dependencies']):
                parts['words'].append(ptb_clean(tok['word']))
                parts['lemmas'].append(ptb_clean(tok['lemma']))
                parts['pos_tags'].append(tok['pos'])
                parts['ner_tags'].append(tok['ner'])
                parts['char_offsets'].append(tok['characterOffsetBegin'])
                dep_par.append(deps['governor'])
                dep_lab.append(deps['dep'])
                dep_order.append(deps['dependent'])

            parts['text'] = ''.join(t['originalText'] + t['after'] for t in block['tokens'])
            
            # make char_offsets relative to start of sentence
            abs_sent_offset = parts['char_offsets'][0]
            parts['char_offsets'] = [p - abs_sent_offset for p in parts['char_offsets']]
            parts['dep_parents'] = sort_X_on_Y(dep_par, dep_order)
            parts['dep_labels'] = sort_X_on_Y(dep_lab, dep_order)
            parts['position'] = position

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
                yield Document(name=name, stable_id=stable_id, text=unicode(text),
                               meta={'file_name' : file_name}), unicode(text)

    def _can_read(self, fpath):
        return fpath.endswith('html') # includes both .html and .xhtml


class SimpleTokenizer:
    """
    A trivial alternative to CoreNLP which parses (tokenizes) text on 
    whitespace only using the split() command.
    """
    def __init__(self, delim):
        self.delim = delim

    def parse(self, document, contents):
        i = 0
        for text in contents.split(self.delim):
            if not len(text.strip()): continue
            words = text.split()
            char_offsets = [0] + list(np.cumsum(map(lambda x: len(x) + 1, words)))[:-1]
            text = ' '.join(words)
            stable_id = construct_stable_id(document, 'phrase', i, i)
            yield {'text': text,
                   'words': words,
                   'char_offsets': char_offsets,
                   'stable_id': stable_id}
            i += 1

class OmniParser(object):
    def __init__(self, 
                 structural=True, blacklist=["style"],              # structural
                 flatten=[], flatten_delim='',   
                 lingual=True, strip=True,                          # lingual
                 replacements=[(u'[\u2010\u2011\u2012\u2013\u2014\u2212\uf02d]','-')],         
                 tabular=True,                                      # tabular
                 visual=False, pdf_path=None):                      # visual
        """
        :param visual: boolean, if True visual features are used in the model
        :param pdf_path: directory where pdf are saved, if a pdf file is not found,
        it will be created from the html document and saved in that directory
        :param replacements: a list of (_pattern_, _replace_) tuples where _pattern_ isinstance
        a regex and _replace_ is a character string. All occurents of _pattern_ in the
        text will be replaced by _replace_.
        """
        self.delim = "<NB>" # NB = New Block

        # structural (html) setup
        self.structural = structural
        self.blacklist = blacklist if isinstance(blacklist, list) else [blacklist]
        self.flatten = flatten if isinstance(flatten, list) else [flatten]
        self.flatten_delim = flatten_delim

        # lingual setup
        self.lingual = lingual
        self.strip = strip
        self.replacements = []
        for (pattern, replace) in replacements:
            self.replacements.append((re.compile(pattern, flags=re.UNICODE), replace))
        if self.lingual:
            self.batch_size = 7000 # TODO: what if this is smaller than a block?
            self.lingual_parse = CoreNLPHandler(delim=self.delim[1:-1]).parse
        else:
            self.batch_size = int(1e6)
            self.lingual_parse = SimpleTokenizer(delim=self.delim).parse

        # tabular setup
        self.tabular = tabular

        # visual setup
        self.visual = visual
        if self.visual:
            self.vizlink = VisualLinker()

    def parse(self, document, text):
        if self.visual:
            if not self.pdf_path:
                warnings.warn("Visual parsing failed: pdf_path is required", RuntimeWarning)
            for _ in self.parse_structure(document, text):
                pass
            # Add visual attributes
            filename = self.pdf_path + document.name
            create_pdf = not os.path.isfile(filename + '.pdf') and not os.path.isfile(filename + '.PDF')
            if create_pdf:  # PDF file does not exist
                self.vizlink.create_pdf(document.name, text)
            for phrase in self.vizlink.parse_visual(document.name, document.phrases, self.pdf_path):
                yield phrase
        else:
            for phrase in self.parse_structure(document, text):
                yield phrase

    def parse_structure(self, document, text):
        self.contents = ""
        block_lengths = []
        self.parent = document

        if self.structural:
            xpaths = []
            html_attrs = []
            html_tags = []

        if self.tabular:
            table_info = TableInfo(document, parent=document)
            self.table_idx = -1
            parents = []
        else:
            table_info = None

        def flatten(node):
            # if a child of this node is in self.flatten, construct a string
            # containing all text/tail results of the tree based on that child
            # and append that to the tail of the previous child or head of node
            num_children = len(node)
            for i, child in enumerate(node[::-1]):
                if child.tag in self.flatten:
                    j = num_children - 1 - i # child index walking backwards
                    contents = ['']
                    for descendant in child.getiterator():
                        if descendant.text and descendant.text.strip():
                            contents.append(descendant.text)
                        if descendant.tail and descendant.tail.strip():
                            contents.append(descendant.tail)
                    if j == 0:
                        if node.text is None:
                            node.text = ''
                        node.text += self.flatten_delim.join(contents)
                    else:
                        if node[j-1].tail is None:
                            node[j-1].tail = ''
                        node[j-1].tail += self.flatten_delim.join(contents)
                    node.remove(child)

        def parse_node(node, table_info=None):
            if node.tag is etree.Comment:
                return
            if self.blacklist and node.tag in self.blacklist:
                return

            if self.tabular:
                self.table_idx = table_info.enter_tabular(node, self.table_idx)

            if self.flatten:
                flatten(node) # flattens children of node that are in the 'flatten' list

            for field in ['text', 'tail']:
                text = getattr(node, field)
                if text is not None:
                    if self.strip:
                        text = text.strip()
                    if len(text):
                        for (rgx, replace) in self.replacements:
                            text = rgx.sub(replace, text)
                        self.contents += text
                        self.contents += self.delim
                        block_lengths.append(len(text) + len(self.delim))
                
                        if self.tabular:
                            parents.append(table_info.parent)
                        
                        if self.structural:
                            context_node = node.getparent() if field=='tail' else node
                            xpaths.append(tree.getpath(context_node))
                            html_tags.append(context_node.tag)
                            html_attrs.append(map(lambda x: '='.join(x), context_node.attrib.items()))
                            
            for child in node:
                if child.tag=='table':
                    parse_node(child, TableInfo(document=table_info.document))
                else:
                    parse_node(child, table_info)
            
            if self.tabular:
                table_info.exit_tabular(node)

        # Parse document and store text in self.contents, padded with self.delim
        root = fromstring(text) # lxml.html.fromstring()
        tree = etree.ElementTree(root)
        document.text = text
        parse_node(root, table_info)
        block_char_end = np.cumsum(block_lengths)

        content_length = len(self.contents)
        parsed = 0
        parent_idx = 0
        position = 0
        phrase_num = 0
        while parsed < content_length:
            batch_end = parsed + \
                        self.contents[parsed:parsed + self.batch_size].rfind(self.delim) + \
                        len(self.delim)
            for parts in self.lingual_parse(document, self.contents[parsed:batch_end]):
                (_, _, _, char_end) = split_stable_id(parts['stable_id'])
                try:
                    while parsed + char_end > block_char_end[parent_idx]:
                        parent_idx += 1
                        position = 0
                    parts['document'] = document
                    parts['phrase_num'] = phrase_num
                    parts['stable_id'] = \
                        "%s::%s:%s:%s" % (document.name, 'phrase', phrase_num, phrase_num)
                    if self.structural:
                        parts['xpath'] =  xpaths[parent_idx]
                        parts['html_tag'] = html_tags[parent_idx]
                        parts['html_attrs'] = html_attrs[parent_idx]
                    if self.tabular:
                        parent = parents[parent_idx]
                        parts = table_info.apply_tabular(parts, parent, position)
                    yield Phrase(**parts) 
                    position += 1
                    phrase_num += 1
                except:
                    import pdb; pdb.set_trace()
            parsed = batch_end

class TableInfo():
    def __init__(self, document,
                 table=None, table_grid=defaultdict(int),
                 cell=None, cell_idx=0,
                 row_idx=0, col_idx=0,
                 parent=None):
        self.document = document
        self.table = table
        self.table_grid = table_grid
        self.cell = cell
        self.cell_idx = cell_idx
        self.row_idx = row_idx
        self.col_idx = col_idx
        self.parent = parent

    def enter_tabular(self, node, table_idx):
        if node.tag == "table":
            table_idx += 1
            self.table_grid.clear()
            self.row_idx = 0
            self.cell_position = 0
            stable_id = "%s::%s:%s:%s" % \
                (self.document.name, "table", table_idx, table_idx)
            self.table = Table(document=self.document, stable_id=stable_id,
                                position=table_idx)
            self.parent = self.table
        elif node.tag == "tr":
            self.col_idx = 0
        elif node.tag in ["td", "th"]:
            # calculate row_start/col_start
            while self.table_grid[(self.row_idx, self.col_idx)]:
                self.col_idx += 1
            col_start = self.col_idx
            row_start = self.row_idx

            # calculate row_end/col_end
            row_end = row_start
            if "rowspan" in node.attrib:
                row_end += int(node.get("rowspan")) - 1
            col_end = col_start
            if "colspan" in node.attrib:
                col_end += int(node.get("colspan")) - 1

            # update table_grid with occupied cells
            for r, c in itertools.product(range(row_start, row_end+1),
                                            range(col_start, col_end+1)):
                self.table_grid[r, c] = 1

            # construct cell
            parts = defaultdict(list)
            parts["document"] = self.document
            parts["table"] = self.table
            parts["row_start"] = row_start
            parts["row_end"] = row_end
            parts["col_start"] = col_start
            parts["col_end"] = col_end
            parts["position"] = self.cell_position
            parts["stable_id"] = "%s::%s:%s:%s:%s" % \
                                    (self.document.name, "cell",
                                     self.table.position, row_start, col_start)
            self.cell = Cell(**parts)
            self.parent = self.cell
        return table_idx

    def exit_tabular(self, node):
        if node.tag == "table":
            self.table = None
            self.parent = self.document
        elif node.tag == "tr":
            self.row_idx += 1
        elif node.tag in ["td", "th"]:
            self.cell = None
            self.col_idx += 1
            self.cell_idx += 1
            self.cell_position += 1
            self.parent = self.table

    def apply_tabular(self, parts, parent, position):
        parts['position'] = position
        if isinstance(parent, Document):
            pass
        elif isinstance(parent, Table):
            parts['table'] = parent
        elif isinstance(parent, Cell):
            parts['table'] = parent.table
            parts['cell'] = parent
            parts['row_start'] = parent.row_start
            parts['row_end'] = parent.row_end
            parts['col_start'] = parent.col_start
            parts['col_end'] = parent.col_end
        else:
            raise NotImplementedError("Phrase parent must be Document, Table, or Cell")
        return parts
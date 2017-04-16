# -*- coding: utf-8 -*-
import atexit
from bs4 import BeautifulSoup
import codecs
from collections import defaultdict
import glob
import json
import lxml.etree as et
import os
import re
import requests
import signal
from subprocess import Popen
import sys
import warnings

from .models import Candidate, Context, Document, Sentence, construct_stable_id
from .udf import UDF, UDFRunner
from .utils import sort_X_on_Y


class CorpusParser(UDFRunner):
    def __init__(self, tok_whitespace=False, split_newline=False, parse_tree=False, fn=None):
        super(CorpusParser, self).__init__(CorpusParserUDF,
                                           tok_whitespace=tok_whitespace,
                                           split_newline=split_newline,
                                           parse_tree=parse_tree,
                                           fn=fn)

    def clear(self, session, **kwargs):
        session.query(Context).delete()
        
        # We cannot cascade up from child contexts to parent Candidates, so we delete all Candidates too
        session.query(Candidate).delete()


class CorpusParserUDF(UDF):
    def __init__(self, tok_whitespace, split_newline, parse_tree, fn, **kwargs):
        self.corenlp_handler = CoreNLPHandler(tok_whitespace=tok_whitespace,
                                              split_newline=split_newline,
                                              parse_tree=parse_tree)
        self.fn = fn
        super(CorpusParserUDF, self).__init__(**kwargs)

    def apply(self, x, **kwargs):
        """Given a Document object and its raw text, parse into processed Sentences"""
        doc, text = x
        for parts in self.corenlp_handler.parse(doc, text):
            parts = self.fn(parts) if self.fn is not None else parts
            yield Sentence(**parts)


class DocPreprocessor(object):
    """
    Processes a file or directory of files into a set of Document objects.

    :param encoding: file encoding to use, default='utf-8'
    :param path: filesystem path to file or directory to parse
    :param max_docs: the maximum number of Documents to produce, default=float('inf')

    """
    def __init__(self, path, encoding="utf-8", max_docs=float('inf')):
        self.path = path
        self.encoding = encoding
        self.max_docs = max_docs

    def generate(self):
        """
        Parses a file or directory of files into a set of Document objects.

        """
        doc_count = 0
        for fp in self._get_files(self.path):
            file_name = os.path.basename(fp)
            if self._can_read(file_name):
                for doc, text in self.parse_file(fp, file_name):
                    yield doc, text
                    doc_count += 1
                    if doc_count >= self.max_docs:
                        return

    def __iter__(self):
        return self.generate()

    def get_stable_id(self, doc_id):
        return "%s::document:0:0" % doc_id

    def parse_file(self, fp, file_name):
        raise NotImplementedError()

    def _can_read(self, fpath):
        return True

    def _get_files(self, path):
        if os.path.isfile(path):
            fpaths = [path]
        elif os.path.isdir(path):
            fpaths = [os.path.join(path, f) for f in os.listdir(path)]
        else:
            fpaths = glob.glob(path)
        if len(fpaths) > 0:
            return fpaths
        else:
            raise IOError("File or directory not found: %s" % (path,))


class TSVDocPreprocessor(DocPreprocessor):
    """Simple parsing of TSV file with one (doc_name <tab> doc_text) per line"""
    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as tsv:
            for line in tsv:
                (doc_name, doc_text) = line.split('\t')
                stable_id=self.get_stable_id(doc_name)
                yield Document(name=doc_name, stable_id=stable_id, meta={'file_name' : file_name}), doc_text


class TextDocPreprocessor(DocPreprocessor):
    """Simple parsing of raw text files, assuming one document per file"""
    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as f:
            name = os.path.basename(fp).rsplit('.', 1)[0]
            stable_id = self.get_stable_id(name)
            yield Document(name=name, stable_id=stable_id, meta={'file_name' : file_name}), f.read()


class CSVPathsPreprocessor(DocPreprocessor):
    """This `DocumentPreprocessor` treats inputs file as index of paths to
     actual documents; each line in the input file contains a path to a document.

     **Defaults and Customization:**

     * The input file is treated as a simple text file having one path per file. However, if the input is a CSV file,
       a pair of ``column`` and ``delim`` parameters may be used to retrieve the desired value as reference path.

     * The referenced documents are treated as text document and hence parsed using ``TextDocPreprocessor``.
       However, if the referenced files are complex, an advanced parser may be used by specifying ``parser_factory``
       parameter to constructor.
     """

    def __init__(self, path, parser_factory=TextDocPreprocessor, column=None,
                 delim=',', *args, **kwargs):
        """
        :param path: input file having paths
        :param parser_factory: The parser class to be used to parse the referenced files.
                                default = TextDocPreprocessor
        :param column: index of the column which references path.
                 default=None, which implies that each line has only one column
        :param delim: delimiter to be used to separate columns when file has
                      more than one column. It is active only when
                      ``column is not None``. default=','
        """
        super(CSVPathsPreprocessor, self).__init__(path, *args, **kwargs)
        self.column = column
        self.delim = delim
        self.parser = parser_factory(path)

    def _get_files(self, path):
        with codecs.open(path, encoding=self.encoding) as lines:
            for doc_path in lines:
                if self.column is not None:
                    # if column is set, retrieve specific column from CSV record
                    doc_path = doc_path.split(self.delim)[self.column]
                yield doc_path.strip()

    def parse_file(self, fp, file_name):
        return self.parser.parse_file(fp, file_name)


class TikaPreprocessor(DocPreprocessor):
    """
    This preprocessor use `Apache Tika <http://tika.apache.org>`_ parser to retrieve text content from
    complex file types such as DOCX, HTML and PDFs.

    Documentation for customizing Tika is `here <https://github.com/chrismattmann/tika-python>`_

    Example::

        !find pdf_dir -name *.pdf > input.csv # list of files
        from snorkel.parser import TikaPreprocessor, CSVPathsPreprocessor, CorpusParser
        CorpusParser().apply(CSVPathsPreprocessor('input.csv', parser_factory=TikaPreprocessor))
    """
    # Tika is conditionally imported here
    import tika
    tika.initVM()   # automatically downloads tika jar and starts a JVM process if no REST API is configured in ENV
    from tika import parser as tk_parser
    parser = tk_parser

    def parse_file(self, fp, file_name):
        parsed = type(self).parser.from_file(fp)
        txt = parsed['content']
        name = os.path.basename(fp).rsplit('.', 1)[0]
        stable_id = self.get_stable_id(name)
        yield Document(name=name, stable_id=stable_id, meta={'file_name': file_name}), txt


class HTMLDocPreprocessor(DocPreprocessor):
    """Simple parsing of raw HTML files, assuming one document per file"""
    def parse_file(self, fp, file_name):
        with open(fp, 'rb') as f:
            html = BeautifulSoup(f, 'lxml')
            txt = filter(self._cleaner, html.findAll(text=True))
            txt = ' '.join(self._strip_special(s) for s in txt if s != '\n')
            name = os.path.basename(fp).rsplit('.', 1)[0]
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


class XMLMultiDocPreprocessor(DocPreprocessor):
    """
    Parse an XML file _which contains multiple documents_ into a set of Document objects.

    Use XPath queries to specify a _document_ object, and then for each document,
    a set of _text_ sections and an _id_.

    **Note: Include the full document XML etree in the attribs dict with keep_xml_tree=True**
    """
    def __init__(self, path, doc='.//document', text='./text/text()', id='./id/text()',
                    keep_xml_tree=False):
        DocPreprocessor.__init__(self, path)
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

PTB = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}', '-LCB-': '{','-RSB-': ']', '-LSB-': '['}

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
        annotators = '"tokenize,ssplit,pos,lemma,depparse,ner{0}"'.format(
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
            warnings.warn("CoreNLP skipped a malformed sentence.", RuntimeWarning)
            return
        position = 0
        for block in blocks:
            parts = defaultdict(list)
            dep_order, dep_par, dep_lab = [], [], []
            for tok, deps in zip(block['tokens'], block['basic-dependencies']):
                # Convert PennTreeBank symbols back into characters for words/lemmas
                parts['words'].append(PTB.get(tok['word'], tok['word']))
                parts['lemmas'].append(PTB.get(tok['lemma'], tok['lemma']))
                parts['pos_tags'].append(tok['pos'])
                parts['ner_tags'].append(tok['ner'])
                parts['char_offsets'].append(tok['characterOffsetBegin'])
                dep_par.append(deps['governor'])
                dep_lab.append(deps['dep'])
                dep_order.append(deps['dependent'])

            parts['text'] = ''.join(t['originalText'] + t.get('after', '') for t in block['tokens'])
            # make char_offsets relative to start of sentence
            abs_sent_offset = parts['char_offsets'][0]
            parts['char_offsets'] = [p - abs_sent_offset for p in parts['char_offsets']]
            parts['dep_parents'] = sort_X_on_Y(dep_par, dep_order)
            parts['dep_labels'] = sort_X_on_Y(dep_lab, dep_order)
            parts['position'] = position

            # Add full dependency tree parse
            if self.parse_tree:
                parts['tree'] = ' '.join(block['parse'].split())

            # Link the sentence to its parent document object
            parts['document'] = document

            # Add null entity array (matching null for CoreNLP)
            parts['entity_cids']  = ['O' for _ in parts['words']]
            parts['entity_types'] = ['O' for _ in parts['words']]

            # Assign the stable id as document's stable id plus absolute character offset
            abs_sent_offset_end = abs_sent_offset + parts['char_offsets'][-1] + len(parts['words'][-1])
            parts['stable_id'] = construct_stable_id(document, 'sentence', abs_sent_offset, abs_sent_offset_end)
            position += 1
            yield parts


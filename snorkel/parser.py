# -*- coding: utf-8 -*-
import atexit
import codecs
import glob
import json
import os
import re
import signal
import sys
import warnings
import requests
import lxml.etree as et

try:
    import spacy
except:
    print>>sys.stderr,"Warning, unable to load 'spaCy' module"

from bs4 import BeautifulSoup
from subprocess import Popen,PIPE
from collections import defaultdict

from .models import Candidate, Context, Document, Sentence, construct_stable_id
from .udf import UDF, UDFRunner
from .utils import sort_X_on_Y


class CorpusParser(UDFRunner):

    def __init__(self, parser=None, fn=None):
        self.parser = StanfordCoreNLPServer() if not parser else parser
        super(CorpusParser, self).__init__(CorpusParserUDF,
                                           parser=self.parser,
                                           fn=fn)
    def clear(self, session, **kwargs):
        session.query(Context).delete()
        # We cannot cascade up from child contexts to parent Candidates, so we delete all Candidates too
        session.query(Candidate).delete()


class CorpusParserUDF(UDF):

    def __init__(self, parser, fn, **kwargs):
        super(CorpusParserUDF, self).__init__(**kwargs)
        self.parser = parser
        self.req_handler = parser.connect()
        self.fn = fn

    def apply(self, x, **kwargs):
        """Given a Document object and its raw text, parse into processed Sentences"""
        doc, text = x
        for parts in self.req_handler.parse(doc, text):
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
                stable_id = self.get_stable_id(doc_name)
                yield Document(name=doc_name, stable_id=stable_id, meta={'file_name': file_name}), doc_text


class TextDocPreprocessor(DocPreprocessor):
    """Simple parsing of raw text files, assuming one document per file"""

    def parse_file(self, fp, file_name):
        with codecs.open(fp, encoding=self.encoding) as f:
            name = os.path.basename(fp).rsplit('.', 1)[0]
            stable_id = self.get_stable_id(name)
            yield Document(name=name, stable_id=stable_id, meta={'file_name': file_name}), f.read()


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
    tika.initVM()  # automatically downloads tika jar and starts a JVM process if no REST API is configured in ENV
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
            yield Document(name=name, stable_id=stable_id, meta={'file_name': file_name}), txt

    def _can_read(self, fpath):
        return fpath.endswith('.html')

    def _cleaner(self, s):
        if s.parent.name in ['style', 'script', '[document]', 'head', 'title']:
            return False
        elif re.match('<!--.*-->', unicode(s)):
            return False
        return True

    def _strip_special(self, s):
        return (''.join(c for c in s if ord(c) < 128)).encode('ascii', 'ignore')


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
        for i, doc in enumerate(et.parse(f).xpath(self.doc)):
            doc_id = str(doc.xpath(self.id)[0])
            text = '\n'.join(filter(lambda t: t is not None, doc.xpath(self.text)))
            meta = {'file_name': str(file_name)}
            if self.keep_xml_tree:
                meta['root'] = et.tostring(doc)
            stable_id = self.get_stable_id(doc_id)
            yield Document(name=doc_id, stable_id=stable_id, meta=meta), text

    def _can_read(self, fpath):
        return fpath.endswith('.xml')


class Parser(object):

    def __init__(self,name):
        self.name = name

    def connect(self):
        '''
        Return connection object for this parser type
        :return:
        '''
        raise NotImplemented

    def close(self):
        '''
        Kill this parser
        :return:
        '''
        raise NotImplemented


class ParserConnection(object):
    '''
    Default connection object assumes local parser object
    '''
    def __init__(self, parser):
        self.parser = parser

    def _connection(self):
        raise NotImplemented

    def parse(self, document, text):
        yield self.parser.parse(document, text)


class URLParserConnection(ParserConnection):
    '''
    URL parser connection
    '''
    def __init__(self, parser):
        self.parser = parser
        self.request = self._connection()

    def _connection(self):
        '''
        Enables retries to cope with CoreNLP server boot-up latency.
        See: http://stackoverflow.com/a/35504626

        Create a new object per connection to make multiprocessing threadsafe.

        :return:
        '''
        from requests.packages.urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        requests_session = requests.Session()
        retries = Retry(total=0,
                        connect=20,
                        read=0,
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504])

        # Mac OS bug -- without this setting multiprocessing requests will fail
        # when the server has boot-up latency associated with model loading
        # See: http://stackoverflow.com/questions/30453152/python-multiprocessing-and-requests
        requests_session.trust_env = False # Don't read proxy settings from OS

        requests_session.mount('http://', HTTPAdapter(max_retries=retries))
        return requests_session

    def parse(self, document, text):
        '''
        Return parse generator
        :param document:
        :param text:
        :return:
        '''
        return self.parser.parse(document, text, self.request)


class SpaCy(Parser):
    '''
    spaCy
    https://spacy.io/

    Minimal (buggy) implementation to show how alternate parsers can be added to Snorkel.
    Models for each target language needs to be downloaded usign the following command:

    python -m spacy download en

    '''
    def __init__(self,lang='en'):

        super(SpaCy, self).__init__(name="spaCy")
        self.model = spacy.load('en')

    def connect(self):
        return ParserConnection(self)

    def parse(self, document, text):
        '''
        Transform spaCy output to match CoreNLP's default format
        :param document:
        :param text:
        :return:
        '''
        if isinstance(text, unicode):
            text = text.encode('utf-8', 'error')
        text = text.decode('utf-8')

        doc = self.model(text)
        assert doc.is_parsed

        position = 0
        for sent in doc.sents:
            parts = defaultdict(list)
            dep_order, dep_par, dep_lab = [], [], []
            for token in sent:
                parts['words'].append(unicode(token))
                parts['lemmas'].append(token.lemma_)
                parts['pos_tags'].append(token.tag_)
                parts['ner_tags'].append(token.ent_type_)
                parts['char_offsets'].append(token.idx)

                dep_par.append(token.head)
                dep_lab.append(token.dep_)
                #dep_order.append(deps['dependent'])

            # Add null entity array (matching null for CoreNLP)
            parts['entity_cids'] = ['O' for _ in parts['words']]
            parts['entity_types'] = ['O' for _ in parts['words']]

            # Link the sentence to its parent document object
            parts['document'] = document
            parts['text'] = sent.text

            # make char_offsets relative to start of sentence
            abs_sent_offset = parts['char_offsets'][0]
            parts['char_offsets'] = [p - abs_sent_offset for p in parts['char_offsets']]
            parts['dep_parents'] = dep_par #sort_X_on_Y(dep_par, dep_order)
            parts['dep_labels'] = dep_lab #sort_X_on_Y(dep_lab, dep_order)
            parts['position'] = position

            # Add full dependency tree parse to document meta
            # TODO

            # Assign the stable id as document's stable id plus absolute character offset
            abs_sent_offset_end = abs_sent_offset + parts['char_offsets'][-1] + len(parts['words'][-1])
            parts['stable_id'] = construct_stable_id(document, 'sentence', abs_sent_offset, abs_sent_offset_end)
            position += 1
            yield parts


class StanfordCoreNLPServer(Parser):
    '''
    Stanford CoreNLP Server

    Implementation uses the simple default web API server
    https://stanfordnlp.github.io/CoreNLP/corenlp-server.html

    Useful configuration examples:

    (1) Disable Penn Treebank Normalization and force strict PTB compliance,
        disabling the following default behaviors:
         (a) Add "." to the end of sentences that end with an abbrv, e.g., Corp.
         (b) Adds a non-breaking space to fractions 5 1/2

        annotator_opts = {}
        annotator_opts['tokenize'] = {"invertible": True,
                                    "normalizeFractions": False,
                                    "normalizeParentheses": False,
                                    "normalizeOtherBrackets": False,
                                    "normalizeCurrency": False,
                                    "asciiQuotes": False,
                                    "latexQuotes": False,
                                    "ptb3Ellipsis": False,
                                    "ptb3Dashes": False,
                                    "escapeForwardSlashAsterisk": False,
                                    "strictTreebank3": True}

    '''
    # Penn TreeBank normalized tokens
    PTB = {'-RRB-': ')', '-LRB-': '(', '-RCB-': '}', '-LCB-': '{', '-RSB-': ']', '-LSB-': '['}

    # CoreNLP changed some JSON element names across versions
    BLOCK_DEFS = {"3.6.0":"basic-dependencies", "3.7.0":"basicDependencies"}

    def __init__(self, annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'depparse', 'ner'],
                 annotator_opts={}, tokenize_whitespace=False, split_newline=False,
                 java_xmx='4g', port=12345, num_threads=1, verbose=False, version='3.6.0'):
        '''
        Create CoreNLP server instance.
        :param annotators:
        :param annotator_opts:
        :param tokenize_whitespace:
        :param split_newline:
        :param java_xmx:
        :param port:
        :param num_threads:
        :param verbose:
        :param version:
        '''
        super(StanfordCoreNLPServer,self).__init__(name="CoreNLP")

        self.tokenize_whitespace = tokenize_whitespace
        self.split_newline = split_newline
        self.annotators = annotators
        self.annotator_opts = annotator_opts

        self.java_xmx = java_xmx
        self.port = port
        self.timeout = 600000
        self.num_threads = num_threads
        self.verbose = verbose
        self.version = version

        # configure connection request options
        opts = self._conn_opts(annotators, annotator_opts, tokenize_whitespace, split_newline)
        self.endpoint = 'http://127.0.0.1:%d/?%s' % (self.port, opts)

        self._start_server()

        if self.verbose:
            self.summary()

    def _start_server(self,force_load=True):
        '''
        Launch CoreNLP server
        :param force_load:  Force server to pre-load models vs. on-demand
        :return:
        '''
        loc = os.path.join(os.environ['SNORKELHOME'], 'parser')
        cmd = 'java -Xmx%s -cp "%s/*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer --port %d --timeout %d --threads %d > /dev/null'
        cmd = [cmd % (self.java_xmx, loc, self.port, self.timeout, self.num_threads)]

        # Setting shell=True returns only the pid of the screen, not any spawned child processes
        # Killing child processes correctly requires using a process group
        # http://stackoverflow.com/questions/4789837/how-to-terminate-a-python-subprocess-launched-with-shell-true
        self.process_group = Popen(cmd, stdout=PIPE, shell=True, preexec_fn=os.setsid)

        if force_load:
            pass

    def _conn_opts(self, annotators, annotator_opts, tokenize_whitespace, split_newline):
        '''
        Server connection properties

        :param annotators:
        :param annotater_opts:
        :param tokenize_whitespace:
        :param split_newline:
        :return:
        '''
        props = [self._get_props(annotators, annotator_opts)]
        if tokenize_whitespace:
            props += ['"tokenize.whitespace": "true"']
        if split_newline and 'ssplit':
            props += ['"ssplit.eolonly": "true"']
        props = ",".join(props)
        return 'properties={%s}' % (props)

    def _get_props(self, annotators, annotator_opts):
        '''
        Enable advanced configuration options for CoreNLP
        Options are configured by each separate annotator

        :param opts: options dictionary
        :return:
        '''
        opts = []
        for name in annotator_opts:
            if not annotator_opts[name]:
                continue
            props = ["{}={}".format(key, str(value).lower()) for key, value in annotator_opts[name].items()]
            opts.append('"{}.options":"{}"'.format(name, ",".join(props)))

        props = []
        props += ['"annotators": {}'.format('"{}"'.format(",".join(annotators)))]
        props += ['"outputFormat": "json"']
        props += [",".join(opts)] if opts else []
        return ",".join(props)

    def summary(self):
        '''
        Print server parameters
        :return:
        '''
        print "------------------------------------"
        print self.endpoint
        print "version:", self.version
        print "shell pid:", self.process_group.pid
        print "port:", self.port
        print "timeout:", self.timeout
        print "threads:", self.num_threads
        print "------------------------------------"

    def connect(self):
        '''
        Return URL connection object for this server
        :return:
        '''
        return URLParserConnection(self)

    def close(self):
        '''
        Kill the process group linked with this server.
        :return:
        '''
        if self.verbose:
            print "Killing CoreNLP server [{}]...".format(self.process_group.pid)
        if self.process_group is not None:
            try:
                os.killpg(os.getpgid(self.process_group.pid), signal.SIGTERM)
            except:
                sys.stderr.write('Could not kill CoreNLP server (might already be closed)...\n'.format(self.process_group.pid))

    def __del__(self):
        '''
        Clean-up this object by forcing the server process to shut-down
        :return:
        '''
        self.close()

    def parse(self, document, text, conn):
        '''
        Parse CoreNLP JSON results. Requires an external connection/request object to remain threadsafe

        :param document:
        :param text:
        :param conn: server URL+properties string
        :return:
        '''
        if len(text.strip()) == 0:
            print>> sys.stderr, "Warning, empty document {0} passed to CoreNLP".format(document.name)
            return

        if isinstance(text, unicode):
            text = text.encode('utf-8', 'error')
        resp = conn.post(self.endpoint, data=text, allow_redirects=True)
        text = text.decode('utf-8')
        content = resp.content.strip()

        # check for parsing error messages
        StanfordCoreNLPServer.validate_response(content)

        try:
            blocks = json.loads(content, strict=False)['sentences']
        except:
            warnings.warn("CoreNLP skipped a malformed sentence.\n{}".format(text), RuntimeWarning)
            return

        position = 0
        for block in blocks:
            parts = defaultdict(list)
            dep_order, dep_par, dep_lab = [], [], []
            for tok, deps in zip(block['tokens'], block[StanfordCoreNLPServer.BLOCK_DEFS[self.version]]):
                # Convert PennTreeBank symbols back into characters for words/lemmas
                parts['words'].append(StanfordCoreNLPServer.PTB.get(tok['word'], tok['word']))
                parts['lemmas'].append(StanfordCoreNLPServer.PTB.get(tok['lemma'], tok['lemma']))
                parts['pos_tags'].append(tok['pos'])
                parts['ner_tags'].append(tok['ner'])
                parts['char_offsets'].append(tok['characterOffsetBegin'])
                dep_par.append(deps['governor'])
                dep_lab.append(deps['dep'])
                dep_order.append(deps['dependent'])

            # certain configuration options remove 'before'/'after' fields in output JSON (TODO: WHY?)
            # In order to create the 'text' field with correct character offsets we use
            # 'characterOffsetEnd' and 'characterOffsetBegin' to build our string from token input
            if not [t for t in block['tokens'] if t.get('after', None)]:
                text = ""
                for t in block['tokens']:
                    # shift to start of local sentence offset
                    i = t['characterOffsetBegin'] - block['tokens'][0]['characterOffsetBegin']
                    # add whitespace based on offsets of originalText
                    text += (' ' * (i - len(text))) + t['originalText'] if len(text) != i else t['originalText']
                parts['text'] = text
            else:
                parts['text'] = ''.join(t['originalText'] + t.get('after', '') for t in block['tokens'])

            # make char_offsets relative to start of sentence
            abs_sent_offset = parts['char_offsets'][0]
            parts['char_offsets'] = [p - abs_sent_offset for p in parts['char_offsets']]
            parts['dep_parents'] = sort_X_on_Y(dep_par, dep_order)
            parts['dep_labels'] = sort_X_on_Y(dep_lab, dep_order)
            parts['position'] = position

            # Add full dependency tree parse to document meta
            if 'parse' in block:  # self.parse_tree:
                tree = ' '.join(block['parse'].split())
                if 'tree' not in document.meta:
                    document.meta['tree'] = {}
                document.meta['tree'][position] = tree

            # Link the sentence to its parent document object
            parts['document'] = document

            # Add null entity array (matching null for CoreNLP)
            parts['entity_cids'] = ['O' for _ in parts['words']]
            parts['entity_types'] = ['O' for _ in parts['words']]

            # Assign the stable id as document's stable id plus absolute character offset
            abs_sent_offset_end = abs_sent_offset + parts['char_offsets'][-1] + len(parts['words'][-1])
            parts['stable_id'] = construct_stable_id(document, 'sentence', abs_sent_offset, abs_sent_offset_end)
            position += 1
            yield parts

    @staticmethod
    def validate_response(content):
        '''
        Report common parsing errors
        :param content:
        :return:
        '''
        if content.startswith("Request is too long"):
            raise ValueError("File too long. Max character count is 100K.")
        if content.startswith("CoreNLP request timed out"):
            raise ValueError("CoreNLP request timed out on file.")
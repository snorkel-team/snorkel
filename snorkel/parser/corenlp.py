# -*- coding: utf-8 -*-
import os
import sys
import json
import signal
import socket
import string
import warnings

from subprocess import Popen,PIPE
from collections import defaultdict

from .parser import Parser, URLParserConnection
from ..models import Candidate, Context, Document, Sentence, construct_stable_id
from ..utils import sort_X_on_Y


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
                 annotator_opts={}, tokenize_whitespace=False, split_newline=False, encoding="utf-8",
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
        super(StanfordCoreNLPServer, self).__init__(name="CoreNLP", encoding=encoding)

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

    def _start_server(self, force_load=False):
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
            conn = self.connect()
            text = "This forces the server to preload all models."
            parts = list(conn.parse(None, text))

    def _conn_opts(self, annotators, annotator_opts, tokenize_whitespace, split_newline):
        '''
        Server connection properties

        :param annotators:
        :param annotater_opts:
        :param tokenize_whitespace:
        :param split_newline:
        :return:
        '''
        # TODO: ssplit options aren't being recognized (but they don't throw any errors either...)
        ssplit_opts = annotator_opts["ssplit"] if "ssplit" in  annotator_opts else {}
        props = [self._get_props(annotators, annotator_opts)]
        if tokenize_whitespace:
            props += ['"tokenize.whitespace": "true"']
        if split_newline:
            props += ['"ssplit.eolonly": "true"']
        if ssplit_opts and 'newlineIsSentenceBreak' in ssplit_opts:
            props += ['"ssplit.newlineIsSentenceBreak": "{}"'.format(ssplit_opts['newlineIsSentenceBreak'])]
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

    def __del__(self):
        '''
        Clean-up this object by forcing the server process to shut-down
        :return:
        '''
        self.close()

    def summary(self):
        '''
        Print server parameters
        :return:
        '''
        print "-" * 40
        print self.endpoint
        print "version:", self.version
        print "shell pid:", self.process_group.pid
        print "port:", self.port
        print "timeout:", self.timeout
        print "threads:", self.num_threads
        print "-" * 40

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
            except Exception as e:
                sys.stderr.write('Could not kill CoreNLP server [{}] {}\n'.format(self.process_group.pid,e))

    def parse(self, document, text, conn):
        '''
        Parse CoreNLP JSON results. Requires an external connection/request object to remain threadsafe

        :param document:
        :param text:
        :param conn: server connection
        :return:
        '''
        if len(text.strip()) == 0:
            print>> sys.stderr, "Warning, empty document {0} passed to CoreNLP".format(document.name if document else "?")
            return

        # handle encoding (force to unicode)
        if isinstance(text, unicode):
            text = text.encode('utf-8', 'error')

        # POST request to CoreNLP Server
        try:
            content = conn.post(self.endpoint, text)
            content = content.decode(self.encoding)

        except socket.error, e:
            print>>sys.stderr,"Socket error"
            raise ValueError("Socket Error")

        # check for parsing error messages
        StanfordCoreNLPServer.validate_response(content)

        try:
            blocks = json.loads(content, strict=False)['sentences']
        except:
            warnings.warn("CoreNLP skipped a malformed document.", RuntimeWarning)

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
            text = ""
            for t in block['tokens']:
                # shift to start of local sentence offset
                i = t['characterOffsetBegin'] - block['tokens'][0]['characterOffsetBegin']
                # add whitespace based on offsets of originalText
                text += (' ' * (i - len(text))) + t['originalText'] if len(text) != i else t['originalText']
            parts['text'] = text

            # make char_offsets relative to start of sentence
            abs_sent_offset = parts['char_offsets'][0]
            parts['char_offsets'] = [p - abs_sent_offset for p in parts['char_offsets']]
            parts['abs_char_offsets'] = [p for p in parts['char_offsets']]
            parts['dep_parents'] = sort_X_on_Y(dep_par, dep_order)
            parts['dep_labels'] = sort_X_on_Y(dep_lab, dep_order)
            parts['position'] = position

            # Add full dependency tree parse to document meta
            if 'parse' in block and document:
                tree = ' '.join(block['parse'].split())
                if 'tree' not in document.meta:
                    document.meta['tree'] = {}
                document.meta['tree'][position] = tree

            # Link the sentence to its parent document object
            parts['document'] = document if document else None

            # Add null entity array (matching null for CoreNLP)
            parts['entity_cids'] = ['O' for _ in parts['words']]
            parts['entity_types'] = ['O' for _ in parts['words']]

            # Assign the stable id as document's stable id plus absolute character offset
            abs_sent_offset_end = abs_sent_offset + parts['char_offsets'][-1] + len(parts['words'][-1])

            if document:
                parts['stable_id'] = construct_stable_id(document, 'sentence', abs_sent_offset, abs_sent_offset_end)
            position += 1
            yield parts

    @staticmethod
    def strip_non_printing_chars(s):
        return "".join([c for c in s if c in string.printable])

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
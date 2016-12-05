'''
Created on Dec 4, 2016

@author: xiao
'''
# -*- coding: utf-8 -*-

from .models import Corpus, Document, Webpage, Sentence, Table, Cell, Phrase, construct_stable_id, split_stable_id
from .utils import ProgressBar, sort_X_on_Y, split_html_attrs
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
from multiprocessing import Pool
from parser import OmniParser
from models.meta import SnorkelSession
from utils import get_ORM_instance

class DocParser:
    """Parse a file into a Document object."""
    def __init__(self, encoding="utf-8"):
        self.encoding = encoding

    def parse(self, fpath):
        """
        Parse a file into a Document object.

        - Input: A file path.
        - Output: A Document object and its text
        """
        if self._can_read(fpath):
            self._parse_file(fpath)

    def get_stable_id(self, doc_id):
        return "%s::document:0:0" % doc_id

    def _parse_file(self, fp):
        raise NotImplementedError()

    def _can_read(self, fpath):
        return True
    
class HTMLParser(DocParser):
    """Simple parsing of files into html documents"""
    def _parse_file(self, fp):
        file_name = os.path.basename(fp)
        with codecs.open(fp, encoding=self.encoding) as f:
            soup = BeautifulSoup(f, 'lxml')
            for text in soup.find_all('html'):
                name = file_name[:file_name.rfind('.')]
                stable_id = self.get_stable_id(name)
                yield Document(name=name, stable_id=stable_id, text=unicode(text),
                               meta={'file_name' : file_name})

    def _can_read(self, fpath):
        return fpath.endswith('html') # includes both .html and .xhtml

def _get_files(path):
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
    
class AsyncOmniParser(OmniParser):
    # TODO move omni parser to async parse and change this API to document only
    # This is just for forcing the evaluation of yield statements
    def parse(self, document):
        for _phrase in super(AsyncOmniParser, self).parse(document, document.text):
            continue

_worker_session = None
_worker_corpus = None
class AsyncParser(object):
    
    def __init__(self, doc_parser, context_parser):
        self.doc_parser = doc_parser
        self.context_parser = context_parser
    
    def parse(self, fpath):
        for document in self.doc_parser.parse(fpath):
            _worker_corpus.add(document)
            self.context_parser.parse(document)
        # Indicate the job is done
        _worker_session.commit()
        return None

def _init_parse_worker(corpus_name):
    '''
    Per process initialization for parsing
    '''
    global _worker_corpus
    global _worker_session
    _worker_session = SnorkelSession()
    _worker_corpus = _worker_session.query(Corpus).filter(Corpus.name==corpus_name).one()

def parse_corpus(session, corpus_name, path, doc_parser, context_parser, max_docs=None, parallel=1):
    fpaths = _get_files(path)
    if max_docs is None: max_docs = len(fpaths)
    fpaths = fpaths[:min(max_docs, len(fpaths))]
    # Actual jobs will assume the shorter of the two lists 
    pb = ProgressBar(len(fpaths))
    # Make sure the corpus exists so that we can add documents to it in workers
    corpus = Corpus(name=corpus_name)
    session.add(corpus)
    session.commit()
    parser = AsyncParser(doc_parser, context_parser)
    tick = 0
    def pb_update(x):
        pb.bar(tick)
        tick += 1
    # Asynchronously parse files        
    pool = Pool(parallel, initializer=_init_parse_worker,initargs=(corpus_name,))
    pool.apply_async(parser.parse, fpaths, callback=pb_update)
    pool.close()
    pool.join()
    pb.close()
    # Load the updated corpus with all documents from workers
    return session.query(Corpus).filter(Corpus.name==corpus_name).one()

# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

import sys
import requests

from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class Parser(object):

    def __init__(self, name, encoding='utf-8'):
        self.name = name
        self.encoding = encoding

    def to_unicode(self, text):
        '''
        Convert char encoding to unicode
        :param text:
        :return:
        '''
        if sys.version_info[0] < 3:
            text_alt = text.encode('utf-8', 'error')
            text_alt = text_alt.decode('string_escape', errors='ignore')
            text_alt = text_alt.decode('utf-8')
            return text_alt
        else:
            return text

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
        return self.parser.parse(document, text)


class URLParserConnection(ParserConnection):
    '''
    URL parser connection
    '''
    def __init__(self, parser, retries=5):
        self.retries = retries
        self.parser = parser
        self.request = self._connection()

    def _connection(self):
        '''
        Enables retries to cope with CoreNLP server boot-up latency.
        See: http://stackoverflow.com/a/35504626

        Create a new object per connection to make multiprocessing threadsafe.

        :return:
        '''
        requests_session = requests.Session()
        retries = Retry(total=self.retries,
                        connect=self.retries,
                        read=self.retries,
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504])

        # Mac OS bug -- without this setting multiprocessing requests will fail
        # when the server has boot-up latency associated with model loading
        # See: http://stackoverflow.com/questions/30453152/python-multiprocessing-and-requests
        if sys.platform in ['darwin']:
            requests_session.trust_env = False
        requests_session.mount('http://', HTTPAdapter(max_retries=retries))
        return requests_session

    def post(self, url, data, allow_redirects=True):
        '''

        :param url:
        :param data:
        :param allow_redirects:
        :param timeout:
        :return:
        '''
        resp = self.request.post(url, data=data, allow_redirects=allow_redirects)
        return resp.content.strip()

    def parse(self, document, text):
        '''
        Return parse generator
        :param document:
        :param text:
        :return:
        '''
        return self.parser.parse(document, text, self)






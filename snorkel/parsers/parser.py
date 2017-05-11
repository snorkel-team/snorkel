# -*- coding: utf-8 -*-
import sys
import requests


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
    def __init__(self, parser, retries=20):
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
        from requests.packages.urllib3.util.retry import Retry
        from requests.adapters import HTTPAdapter
        requests_session = requests.Session()
        retries = Retry(total=self.retries,
                        connect=20,
                        read=0,
                        backoff_factor=0.1,
                        status_forcelist=[500, 502, 503, 504])

        # Mac OS bug -- without this setting multiprocessing requests will fail
        # when the server has boot-up latency associated with model loading
        # See: http://stackoverflow.com/questions/30453152/python-multiprocessing-and-requests
        if sys.platform in ['darwin']:
            requests_session.trust_env = False
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






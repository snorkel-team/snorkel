import sys
import requests

from .corenlp import StanfordCoreNLPServer
from ..models import Candidate, Context, Sentence
from ..udf import UDF, UDFRunner


class CorpusParser(UDFRunner):

    def __init__(self, parser=None, fn=None):
        self.parser = StanfordCoreNLPServer() if not parser else parser
        super(CorpusParser, self).__init__(CorpusParserUDF,
                                           parser=self.parser,
                                           fn=fn)
    def clear(self, session, **kwargs):
        session.query(Context).delete()
        # We cannot cascade up from child contexts to parent Candidates,
        # so we delete all Candidates too
        session.query(Candidate).delete()


class CorpusParserUDF(UDF):

    def __init__(self, parser, fn, **kwargs):
        super(CorpusParserUDF, self).__init__(**kwargs)
        self.parser = parser
        self.req_handler = parser.connect()
        self.fn = fn

    def apply(self, x, **kwargs):
        """Given a Document object and its raw text, parse into Sentences"""
        doc, text = x
        for parts in self.req_handler.parse(doc, text):
            parts = self.fn(parts) if self.fn is not None else parts
            yield Sentence(**parts)


class Parser(object):

    def __init__(self,name):
        self.name = name

    def connect(self):
        '''
        Return connection object for this parser type
        :return:
        '''
        raise NotImplementedError()

    def close(self):
        '''
        Kill this parser
        :return:
        '''
        raise NotImplementedError()


class ParserConnection(object):
    '''
    Default connection object assumes local parser object
    '''
    def __init__(self, parser):
        self.parser = parser

    def _connection(self):
        raise NotImplementedError()

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
        # See: http://stackoverflow.com/questions/30453152
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

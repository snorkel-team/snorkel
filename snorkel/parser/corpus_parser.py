from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

from .corenlp import StanfordCoreNLPServer
from ..models import Candidate, Context, Sentence, Span
from ..udf import UDF, UDFRunner, UDFRunnerBatches


class CorpusParser(UDFRunner):

    def __init__(self, parser=None, fn=None):
        self.parser = parser or StanfordCoreNLPServer()
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


class BatchFilter(object):
    def __init__(self,session = None,**kwargs):
        super(BatchFilter, self).__init__(**kwargs)
        self.session = session

    def filter_batch(self,batch_dico):
        new_batch_dico = dict()
        for k in batch_dico.keys():
            if self.filter_item(batch_dico[k]):
                new_batch_dico[k] = batch_dico[k]
        return new_batch_dico

    def filter_item(self,batch_item):
        raise NotImplementedError()

    def clean_batch(self):
        raise NotImplementedError()

class CEFilter(BatchFilter):

    def __init__(self,CExtractor,**kwargs):
        super(CEFilter,self).__init__(**kwargs)
        self.cand_extractor = CExtractor

    def filter_item(self,batch_item):

        udf = self.cand_extractor.udf_class(**self.cand_extractor.udf_init_kwargs)
        if self.session:
            udf.session.close()
            udf.session = self.session
        items = list()
        for item in batch_item:
            items.extend(list(udf.apply(item,clear = True,split = 0,return_type = "dummy")))
        #print(len(items))
        return len(items) > 0

    def clean_batch(self):
        self.session.query(Context).filter(Context.type=="span").delete()
        self.session.query(Span).delete()
        self.session.commit()

class TrivialFilter(BatchFilter):

    def filter_item(self,batch_item):
        return True

    def clean_batch(self):
        pass



class CorpusParserFilter(UDFRunnerBatches):

    def __init__(self,batch_filter, udf_batch_size = 10 ,parser=None, fn=None):
        self.parser = parser or StanfordCoreNLPServer()
        super(CorpusParserFilter, self).__init__(CorpusParserUDF,
                                           parser=self.parser,
                                           fn=fn)
        self.udf_batch_size=udf_batch_size
        self.batch_filter = batch_filter

    def clear(self, session, **kwargs):
        session.query(Context).delete()
        # We cannot cascade up from child contexts to parent Candidates,
        # so we delete all Candidates too
        session.query(Candidate).delete()
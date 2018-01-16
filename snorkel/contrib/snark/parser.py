from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import *

class SparkCorpusParser(object):
    """
    Distributes raw documents to a Spark cluster and applies a parser to them,
    returning a hierarchy of Context objects. See snorkel.parser.CorpusParser.

    NOTE: Currently just a stub.
    """
    def __init__(self, snorkel_session, spark_session, parser, fn=None):
        """
        Constructor

        :param snorkel_session: the SnorkelSession for the Snorkel application
        :param spark_session: a PySpark SparkSession
        :param parser: See snorkel.parser.Parser
        :param fn: Function to apply to parser output
        """
        self.snorkel_session = snorkel_session
        self.spark_session = spark_session
        self.parser = parser
        self.fn = fn

    def apply(self, docs, **kwargs):
        # TODO: Need to be able to start a Parser server on each server...
        pass

    def _clear(self):
        # TODO
        # session.query(Context).delete()
        # # We cannot cascade up from child contexts to parent Candidates,
        # # so we delete all Candidates too
        # session.query(Candidate).delete()
        pass
"""Extracting symptoms from diseases
"""
from snorkel import SnorkelSession
from snorkel.parser import XMLMultiDocPreprocessor
from snorkel.parser import CorpusParser
from metamap import MetaMapAPI
from snorkel.models import Document, Sentence

data_file_path = 'tutorials/cdr/data/CDR.BioC.xml'

SNORKEL_SESSION = SnorkelSession()

doc_preprocessor = XMLMultiDocPreprocessor(
    path=data_file_path,
    doc='.//document',
    text='.//passage/text/text()',
    id='.//id/text()'
)

metamap_api = MetaMapAPI()
corpus_parser = CorpusParser(fn=metamap_api.tag)
corpus_parser.apply(list(doc_preprocessor))

print("Documents:", SNORKEL_SESSION.query(Document).count())
print("Sentences:", SNORKEL_SESSION.query(Sentence).count())

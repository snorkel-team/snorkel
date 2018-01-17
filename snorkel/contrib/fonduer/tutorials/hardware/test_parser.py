import os
import sys

PARALLEL = 1  # assuming a quad-core machine
ATTRIBUTE = "stg_temp_max"

os.environ['FONDUERDBNAME'] = ATTRIBUTE
os.environ['SNORKELDB'] = 'postgres://localhost:5432/' + os.environ['FONDUERDBNAME']

from snorkel.contrib.fonduer import SnorkelSession

session = SnorkelSession()

from snorkel.parser.spacy_parser import Spacy
from snorkel.contrib.fonduer import HTMLPreprocessor, OmniParser

docs_path = os.environ['FONDUERHOME'] + '/tutorials/hardware/data/html_TreeStructure/'
pdf_path = os.environ['FONDUERHOME'] + '/tutorials/hardware/data/pdf_TreeStructure/'

max_docs = 6
doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)

#  corpus_parser = OmniParser(structural=True, lingual=True, visual=False, pdf_path=pdf_path)
corpus_parser = OmniParser(structural=True, lingual=True, visual=False, pdf_path=pdf_path, lingual_parser=Spacy())
corpus_parser.apply(doc_preprocessor)  # , parallelism=PARALLEL)

from snorkel.contrib.fonduer.models import Document, Phrase, Figure, Para

print "Documents:", session.query(Document).count()
print "Phrases:", session.query(Phrase).count()
print "Paragraphs:", session.query(Para).count()
print "Figures:", session.query(Figure).count()

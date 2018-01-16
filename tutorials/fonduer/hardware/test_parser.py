import os
import sys

PARALLEL = 4 # assuming a quad-core machine
ATTRIBUTE = "stg_temp_max"

os.environ['SNORKELDBNAME'] = ATTRIBUTE
os.environ['SNORKELDB'] = 'postgres://pabajaj:123@localhost:5433/' + os.environ['SNORKELDBNAME']

sys.path.append(os.environ['SNORKELHOME'] + '/tutorials/fonduer/hardware/')

from snorkel.contrib.fonduer import SnorkelSession

session = SnorkelSession()

from snorkel.contrib.fonduer import HTMLPreprocessor, OmniParser

docs_path = os.environ['SNORKELHOME'] + '/tutorials/fonduer/hardware/data/html_TreeStructure/'
pdf_path = os.environ['SNORKELHOME'] + '/tutorials/fonduer/hardware/data/pdf_TreeStructure/'

max_docs = 6
doc_preprocessor = HTMLPreprocessor(docs_path, max_docs=max_docs)

corpus_parser = OmniParser(structural=True, lingual=True, visual=False, pdf_path=pdf_path)
corpus_parser.apply(doc_preprocessor, parallelism=PARALLEL)

from snorkel.contrib.fonduer.models import Document, Phrase, Figure, Para

print "Documents:", session.query(Document).count()
print "Phrases:", session.query(Phrase).count()
print "Paragraphs:", session.query(Para).count()
print "Figures:", session.query(Figure).count()

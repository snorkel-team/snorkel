"""
Subpackage for all built-in Snorkel data models.

To ensure correct behavior, this subpackage should always be treated as a single module (with one exception
described below). This rule means that all data models should be imported from this subpackage,
not directly from individual submodules. For example, the correct way to import the Corpus class is
>>> from snorkel.models import Corpus

The only exception is importing from snorkel.models.meta in order to extend Snorkel's data models.
"""
from meta import SnorkelBase, SnorkelSession, snorkel_engine, snorkel_postgres
from corpus import Corpus, Document, Context, Sentence
from candidates import Candidates, Candidate, Ngram

# This call must be performed after all classes that extend SnorkelBase are
# declared to ensure the database schema is initialized
SnorkelBase.metadata.create_all(snorkel_engine)

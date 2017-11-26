from .....models.meta import SnorkelSession, SnorkelBase, snorkel_engine
from .....models.context import Document
from .....models.candidate import candidate_subclass

from .context import Webpage, Table, Cell, Phrase, Figure, TemporaryImplicitSpan, ImplicitSpan, TemporaryImage, Image

# Use sqlalchemy to create tables for the new context types used by Fonduer
SnorkelBase.metadata.create_all(snorkel_engine)

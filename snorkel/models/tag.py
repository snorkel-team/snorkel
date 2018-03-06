from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from sqlalchemy import (
    Column, String, Integer, Float, Boolean, ForeignKey, UniqueConstraint,
    MetaData
)

from .meta import SnorkelBase
from ..models import snorkel_engine
from ..utils import camel_to_under

class SequenceTag(SnorkelBase):
    """
    This table stores sequence tags from provided by NER taggers.
    Tags are defined by document-level absolute character offsets.
    """
    __tablename__ = 'sequence_tag'
    id = Column(Integer, primary_key=True)
    document_id    = Column(Integer, ForeignKey('document.id'))
    abs_char_start = Column(Integer, nullable=False)
    abs_char_end   = Column(Integer, nullable=False)
    concept_type   = Column(String, nullable=False) # , index=True
    concept_uid    = Column(String, nullable=True)
    source         = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint(document_id, abs_char_start, abs_char_end, concept_type, concept_uid, source),
    )

    def __repr__(self):
        return "<SequenceTag: {}({}:{}-{} [{}])>".format(self.concept_type, self.document_id,
                                                         self.abs_char_start, self.abs_char_end,
                                                         self.concept_uid if self.concept_uid else "NONE")





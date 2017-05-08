from .....models.meta import SnorkelBase, SnorkelSession, snorkel_engine, snorkel_postgres
from .....models.context import construct_stable_id, split_stable_id
from .....models.candidate import Candidate, candidate_subclass
from .....models.annotation import Feature, FeatureKey, Label, LabelKey, GoldLabel, GoldLabelKey, StableLabel, Prediction, PredictionKey
from .....models.parameter import Parameter
from .context import Webpage, Table, Cell, Phrase, TemporaryImplicitSpan, ImplicitSpan, Document


# SnorkelBase is initialized by snorkel.models.
#
# Now, we want to add tables for the new context types used by Fonduer, and
# override some of the default tables (e.g. Document) with new fields.
# SnorkelBase.metadata.drop_all(snorkel_engine)
SnorkelBase.metadata.create_all(snorkel_engine)

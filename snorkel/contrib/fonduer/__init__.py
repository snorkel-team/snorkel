import sys

# If we want to directly expose any subclasses at the package level, include
# them here. For example, this allows the user to write
# `from snorkel.contrib.fonduer import HTMLPreprocessor` rather than having
# `from snorkel.contrib.fonduer.fonduer.parser import HTMLPreprocessor`
from .fonduer import SnorkelSession
from .fonduer.parser import HTMLPreprocessor, OmniParser
from .fonduer.async_annotations import BatchFeatureAnnotator, BatchLabelAnnotator

# Raise the visibility of these subpackages to the package level for cleaner
# syntax. The key idea here is when we do `from package.submodule1 import foo`
# `sys.modules` is checked to see if it has package. If not, then
# `package/__init__.py` is run and loaded. Then, the process repeats for
# `package.submodule1`. We can omit the fonduer submodule in the path by
# using this __init__.py to put these packages in sys.modules directly.
#
# This allows `from snorkel.contrib.fonduer.models import Phrase` rather than
# need to write `from snorkel.contrib.fonduer.fonduer.models import Phrase`
from .fonduer import models
from .fonduer import lf_helpers
from .fonduer import visualizer
from .fonduer import candidates

for module in [models, lf_helpers, visualizer, candidates]:
    full_name = '{}.{}'.format(__package__, module.__name__.rsplit('.')[-1])
    sys.modules[full_name] = sys.modules[module.__name__]

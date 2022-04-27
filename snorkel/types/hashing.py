from collections.abc import Hashable
from typing import Any, Callable

HashingFunction = Callable[[Any], Hashable]

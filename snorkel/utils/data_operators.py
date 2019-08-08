from collections import Counter
from typing import List


def check_unique_names(names: List[str]) -> None:
    """Check that operator names are unique."""
    k, ct = Counter(names).most_common(1)[0]
    if ct > 1:
        raise ValueError(f"Operator names not unique: {ct} operators with name {k}")

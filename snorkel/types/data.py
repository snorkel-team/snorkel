from typing import Any, Mapping, Sequence, Union

import numpy as np
import scipy.sparse as sparse
from torch import Tensor

DataPoint = Any
DataPoints = Sequence[DataPoint]

Field = Any
FieldMap = Mapping[str, Field]

ArrayLike = Union[np.ndarray, list, sparse.spmatrix, Tensor]

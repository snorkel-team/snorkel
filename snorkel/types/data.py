from typing import Any, Collection, Mapping, Union

import numpy as np
import scipy.sparse as sparse
import torch

DataPoint = Any
DataPoints = Collection[DataPoint]

Field = Any
FieldMap = Mapping[str, Field]

ArrayLike = Union[np.ndarray, list, sparse.spmatrix, torch.Tensor]

#!/bin/env python3

from typing import List
import numpy as np
from ._pycondec import cond_jit

@cond_jit(nopython=True,cache=True)
def dots(mats: List[np.ndarray]) -> np.ndarray:
    """Left-to-right chained matrix product ``mats[0] @ mats[1] @ ... @ mats[-1]``.

    This is a *chained 2-D matrix product*, not a batched/broadcasting matmul.
    Under numba (the compiled path) ``np.dot`` supports only 2-D operands, so:

    - ``mats`` must be a non-empty, homogeneous list of 2-D ndarrays
      (float or complex) with matmul-compatible inner dimensions.
    - Stacked/broadcasted inputs (e.g. a list of ``(N, 3, 3)`` arrays) are NOT
      supported and raise a numba ``TypingError``. For batched matrix products
      use ``np.matmul`` (which broadcasts) instead.

    Args:
        mats: Non-empty list of 2-D matrices to multiply from left to right.

    Returns:
        The 2-D matrix product of all entries.
    """
    mat = mats[0]
    for i in range(1, len(mats)):
        mat = np.dot(mat, mats[i])
    return mat

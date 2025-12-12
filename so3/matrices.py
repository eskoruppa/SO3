#!/bin/env python3

from typing import List
import numpy as np
from .pyConDec.pycondec import cond_jit

@cond_jit(nopython=True,cache=True)
def dots(mats: List[np.ndarray]) -> np.ndarray:
    mat = mats[0]
    for i in range(1, len(mats)):
        mat = np.dot(mat, mats[i])
    return mat

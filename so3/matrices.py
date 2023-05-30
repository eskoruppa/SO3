#!/bin/env python3

import numpy as np
from typing import List
from .pyConDec.pycondec import cond_jit


def dots(mats: List[np.ndarray]) -> np.ndarray:
    mat = mats[0]
    for i in range(1,len(mats)):
        mat = np.dot(mat,mats[i])
    return mat
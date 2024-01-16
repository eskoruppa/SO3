#!/bin/env python3

import numpy as np
from typing import List
from .pyConDec.pycondec import cond_jit

@cond_jit
def se3_inverse(g: np.ndarray) -> np.ndarray:
    inv = np.zeros(g.shape)
    inv[:3,:3] = g[:3,:3].T
    inv[:3,3]  = -inv[:3,:3]@g[:3,3]
    inv[3,3] = 1
    return inv
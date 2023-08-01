#!/bin/env python3

import numpy as np
from .pyConDec.pycondec import cond_jit

from .generators import hat_map, vec_map #, generator1, generator2, generator3

@cond_jit
def cayley2rotmat(cayley: np.ndarray) -> np.ndarray:
    """Transforms cayley vector to corresponding rotation matrix

    Args:
        cayley (np.ndarray): Cayley vector

    Returns:
        np.ndarray: rotation matrix
    """
    hat = hat_map(cayley)
    return np.eye(3) + 4./(4+np.dot(cayley,cayley)) * (hat + 0.5*np.dot(hat,hat))
    
@cond_jit
def rotmat2cayley(rotmat: np.ndarray) -> np.ndarray:
    """Transforms rotation matrix to corresponding Cayley vector

    Args:
        rotmat (np.ndarray): element of SO(3)

    Returns:
        np.ndarray: returns 3-vector
    """
    return 2./(1+np.trace(rotmat)) * vec_map(rotmat-rotmat.T)

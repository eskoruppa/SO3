#!/bin/env python3

import numpy as np

from ._pycondec import cond_jit


##########################################################################################################
# Internal single-vector JIT helpers
# Defined first so they remain callable from within other JIT contexts.
##########################################################################################################

@cond_jit(nopython=True, cache=True)
def _hat_map_sv(x: np.ndarray) -> np.ndarray:
    """Maps a single rotation vector onto the corresponding so(3) element. JIT-callable."""
    X = np.zeros((3, 3))
    X[0, 1] = -x[2]
    X[1, 0] = x[2]
    X[0, 2] = x[1]
    X[2, 0] = -x[1]
    X[1, 2] = -x[0]
    X[2, 1] = x[0]
    return X


@cond_jit(nopython=True, cache=True)
def _vec_map_sv(X: np.ndarray) -> np.ndarray:
    """Extracts the rotation vector from a single so(3) element. JIT-callable."""
    return np.array([X[2, 1], X[0, 2], X[1, 0]])


##########################################################################################################
# Internal flat-batch JIT helpers
##########################################################################################################

@cond_jit(nopython=True, cache=True)
def _hat_map_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3, 3))
    for i in range(n):
        result[i] = _hat_map_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _vec_map_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3))
    for i in range(n):
        result[i] = _vec_map_sv(flat[i])
    return result


##########################################################################################################
# Public batch dispatchers
##########################################################################################################

def hat_map_batch(x: np.ndarray) -> np.ndarray:
    """Maps rotation vectors (Euler vectors) onto the corresponding elements of so(3).

    Accepts any shape (..., 3). Output shape: (..., 3, 3).
    For a single vector (3,) returns a (3, 3) matrix.

    Args:
        x (np.ndarray): Euler vector(s), last dimension must be 3.

    Returns:
        np.ndarray: so(3) element(s), shape (..., 3, 3).
    """
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return _hat_map_sv(x)
    orig_shape = x.shape
    n = x.size // 3
    flat = np.ascontiguousarray(x).reshape((n, 3))
    return _hat_map_flat(flat).reshape(orig_shape[:-1] + (3, 3))


def vec_map_batch(X: np.ndarray) -> np.ndarray:
    """Inverse of the hat map. Maps elements of so(3) onto the corresponding Euler vectors.

    A single matrix has shape (3, 3); a batch has shape (..., 3, 3).
    Output shape: (..., 3), or (3,) for a single matrix.

    Args:
        X (np.ndarray): so(3) element(s), last two dimensions must be (3, 3).

    Returns:
        np.ndarray: rotation vector(s), shape (..., 3).
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 2:
        return _vec_map_sv(X)
    orig_shape = X.shape
    n = X.size // 9
    flat = np.ascontiguousarray(X).reshape((n, 3, 3))
    return _vec_map_flat(flat).reshape(orig_shape[:-2] + (3,))


##########################################################################################################
# Generators of SO(3) 
##########################################################################################################

@cond_jit(nopython=True, cache=True)
def generator1() -> np.ndarray:
    """first generator of SO(3)

    Returns:
        np.ndarray: L1
    """
    X = np.zeros((3, 3))
    X[1, 2] = -1
    X[2, 1] = 1
    return X


@cond_jit(nopython=True, cache=True)
def generator2() -> np.ndarray:
    """second generator of SO(3)

    Returns:
        np.ndarray: L2
    """
    X = np.zeros((3, 3))
    X[0, 2] = 1
    X[2, 0] = -1
    return X


@cond_jit(nopython=True, cache=True)
def generator3() -> np.ndarray:
    """third generator of SO(3)

    Returns:
        np.ndarray: L3
    """
    X = np.zeros((3, 3))
    X[0, 1] = -1
    X[1, 0] = 1
    return X


##########################################################################################################
# Public single-vector API
# Original implementations kept verbatim under _single names. Also aliased from _sv helpers.
##########################################################################################################

@cond_jit(nopython=True, cache=True)
def hat_map_single(x: np.ndarray) -> np.ndarray:
    """Maps a single rotation vector (Euler vector) onto the corresponding element of so(3).

    This is the original single-input implementation. JIT-callable.

    Args:
        x (np.ndarray): euler vector (3-vector)

    Returns:
        np.ndarray: rotation matrix (element of so(3))
    """
    X = np.zeros((3, 3))
    X[0, 1] = -x[2]
    X[1, 0] = x[2]
    X[0, 2] = x[1]
    X[2, 0] = -x[1]
    X[1, 2] = -x[0]
    X[2, 1] = x[0]
    return X


@cond_jit(nopython=True, cache=True)
def vec_map_single(X: np.ndarray) -> np.ndarray:
    """Inverse of the hat map for a single so(3) element (3, 3) → (3,).

    This is the original single-input implementation. JIT-callable.

    Args:
        X (np.ndarray): generator of SO(3) (element of so(3)). Skew-symmetric matrix.

    Returns:
        np.ndarray: rotation vector (3-vector)
    """
    return np.array([X[2, 1], X[0, 2], X[1, 0]])
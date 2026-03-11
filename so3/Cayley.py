#!/bin/env python3

import numpy as np

from .generators import _hat_map_sv, _vec_map_sv
from .generators import hat_map_single, vec_map_single
from ._pycondec import cond_jit

##########################################################################################################
############### SO3 Methods ##############################################################################
##########################################################################################################

##########################################################################################################
# Internal single-vector JIT helpers
##########################################################################################################

@cond_jit(nopython=True, cache=True)
def _cayley2rotmat_sv(cayley: np.ndarray) -> np.ndarray:
    hat = _hat_map_sv(cayley)
    return np.eye(3) + 4.0 / (4 + np.dot(cayley, cayley)) * (
        hat + 0.5 * np.dot(hat, hat)
    )


@cond_jit(nopython=True, cache=True)
def _rotmat2cayley_sv(rotmat: np.ndarray) -> np.ndarray:
    denom = 1.0 + np.trace(rotmat)
    if np.abs(denom) < 1e-10:
        # trace(R) = -1 means theta = pi: Cayley map is singular here.
        # Return a vector of infinities to signal the singularity.
        return np.array([np.inf, np.inf, np.inf])
    return 2.0 / denom * _vec_map_sv(rotmat - rotmat.T)


@cond_jit(nopython=True, cache=True)
def _se3_cayley2rotmat_sv(Omega: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    if rotation_first:
        vrot = Omega[:3]
        vtrans = Omega[3:]
    else:
        vrot = Omega[3:]
        vtrans = Omega[:3]
    rotmat = np.zeros((4, 4))
    rotmat[:3, :3] = _cayley2rotmat_sv(vrot)
    rotmat[:3, 3] = vtrans
    rotmat[3, 3] = 1
    return rotmat


@cond_jit(nopython=True, cache=True)
def _se3_rotmat2cayley_sv(R: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    vrot = _rotmat2cayley_sv(R[:3, :3])
    vtrans = R[:3, 3]
    if rotation_first:
        return np.concatenate((vrot, vtrans))
    else:
        return np.concatenate((vtrans, vrot))


##########################################################################################################
# Internal flat-batch JIT helpers
##########################################################################################################

@cond_jit(nopython=True, cache=True)
def _cayley2rotmat_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3, 3))
    for i in range(n):
        result[i] = _cayley2rotmat_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _rotmat2cayley_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3))
    for i in range(n):
        result[i] = _rotmat2cayley_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _se3_cayley2rotmat_flat(flat: np.ndarray, rotation_first: bool) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 4, 4))
    for i in range(n):
        result[i] = _se3_cayley2rotmat_sv(flat[i], rotation_first)
    return result


@cond_jit(nopython=True, cache=True)
def _se3_rotmat2cayley_flat(flat: np.ndarray, rotation_first: bool) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 6))
    for i in range(n):
        result[i] = _se3_rotmat2cayley_sv(flat[i], rotation_first)
    return result


##########################################################################################################
# Public batch dispatchers
##########################################################################################################

def cayley2rotmat_batch(cayley: np.ndarray) -> np.ndarray:
    """Transforms Cayley vector(s) to the corresponding rotation matri(x/ces).

    Accepts any shape (..., 3). Output shape: (..., 3, 3).
    For a single vector (3,) returns a (3, 3) matrix.

    Args:
        cayley (np.ndarray): Cayley vector(s), last dimension must be 3.

    Returns:
        np.ndarray: rotation matrix/matrices, shape (..., 3, 3).
    """
    cayley = np.asarray(cayley, dtype=float)
    if cayley.ndim == 1:
        return _cayley2rotmat_sv(cayley)
    orig_shape = cayley.shape
    n = cayley.size // 3
    flat = np.ascontiguousarray(cayley).reshape((n, 3))
    return _cayley2rotmat_flat(flat).reshape(orig_shape[:-1] + (3, 3))

def rotmat2cayley_batch(rotmat: np.ndarray) -> np.ndarray:
    """Transforms rotation matri(x/ces) to the corresponding Cayley vector(s).

    A single matrix has shape (3, 3); a batch has shape (..., 3, 3).
    Output shape: (..., 3), or (3,) for a single matrix.

    Args:
        rotmat (np.ndarray): rotation matrix/matrices, last two dimensions must be (3, 3).

    Returns:
        np.ndarray: Cayley vector(s), shape (..., 3).
    """
    rotmat = np.asarray(rotmat, dtype=float)
    if rotmat.ndim == 2:
        return _rotmat2cayley_sv(rotmat)
    orig_shape = rotmat.shape
    n = rotmat.size // 9
    flat = np.ascontiguousarray(rotmat).reshape((n, 3, 3))
    return _rotmat2cayley_flat(flat).reshape(orig_shape[:-2] + (3,))


##########################################################################################################
############### SE3 Methods ##############################################################################
##########################################################################################################

def se3_cayley2rotmat_batch(Omega: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Transforms SE(3) Cayley 6-vector(s) to 4×4 SE(3) matri(x/ces).

    Accepts shape (6,) or (..., 6). Output shape: (4, 4) or (..., 4, 4).

    Args:
        Omega (np.ndarray): SE(3) Cayley vector(s).
        rotation_first (bool): if True, first 3 components are rotation.

    Returns:
        np.ndarray: SE(3) matrix/matrices, shape (..., 4, 4).
    """
    Omega = np.asarray(Omega, dtype=float)
    if Omega.ndim == 1:
        return _se3_cayley2rotmat_sv(Omega, rotation_first)
    orig_shape = Omega.shape
    n = Omega.size // 6
    flat = np.ascontiguousarray(Omega).reshape((n, 6))
    return _se3_cayley2rotmat_flat(flat, rotation_first).reshape(orig_shape[:-1] + (4, 4))


def se3_rotmat2cayley_batch(R: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Transforms SE(3) matri(x/ces) to 6-vector(s) in Cayley parametrization.

    A single matrix has shape (4, 4); a batch has shape (..., 4, 4).
    Output shape: (6,) or (..., 6).

    Args:
        R (np.ndarray): SE(3) matrix/matrices, last two dimensions must be (4, 4).
        rotation_first (bool): if True, first 3 components are rotation.

    Returns:
        np.ndarray: Cayley SE(3) vector(s), shape (..., 6).
    """
    R = np.asarray(R, dtype=float)
    if R.ndim == 2:
        return _se3_rotmat2cayley_sv(R, rotation_first)
    orig_shape = R.shape
    n = R.size // 16
    flat = np.ascontiguousarray(R).reshape((n, 4, 4))
    return _se3_rotmat2cayley_flat(flat, rotation_first).reshape(orig_shape[:-2] + (6,))


##########################################################################################################
# Public single-vector API
# Original implementations kept verbatim under _single names. JIT-callable.
##########################################################################################################


@cond_jit(nopython=True,cache=True)
def cayley2rotmat_single(cayley: np.ndarray) -> np.ndarray:
    """Transforms cayley vector to corresponding rotation matrix

    Args:
        cayley (np.ndarray): Cayley vector

    Returns:
        np.ndarray: rotation matrix
    """
    hat = hat_map_single(cayley)
    return np.eye(3) + 4.0 / (4 + np.dot(cayley, cayley)) * (
        hat + 0.5 * np.dot(hat, hat)
    )


@cond_jit(nopython=True,cache=True)
def rotmat2cayley_single(rotmat: np.ndarray) -> np.ndarray:
    """Transforms rotation matrix to corresponding Cayley vector

    Args:
        rotmat (np.ndarray): element of SO(3). Must not be a 180-degree rotation
            (theta=pi), where the Cayley parametrization is singular.

    Returns:
        np.ndarray: returns 3-vector (inf if theta=pi)
    """
    denom = 1.0 + np.trace(rotmat)
    if np.abs(denom) < 1e-10:
        # trace(R) = -1 means theta = pi: Cayley map is singular here.
        return np.array([np.inf, np.inf, np.inf])
    return 2.0 / denom * vec_map_single(rotmat - rotmat.T)


def se3_cayley2rotmat_single(Omega: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    if Omega.shape != (6,):
        raise ValueError(f"Expected shape (6,) array, but encountered {Omega.shape}.")
    if rotation_first:
        vrot = Omega[:3]
        vtrans = Omega[3:]
    else:
        vrot = Omega[3:]
        vtrans = Omega[:3]
    rotmat = np.zeros((4, 4))
    rotmat[:3, :3] = cayley2rotmat_single(vrot)
    rotmat[:3, 3] = vtrans
    rotmat[3, 3] = 1
    return rotmat


def se3_rotmat2cayley_single(R: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    if R.shape != (4, 4):
        raise ValueError(f"Expected shape (4,4) array, but encountered {R.shape}.")
    vrot = rotmat2cayley_single(R[:3, :3])
    vtrans = R[:3, 3]
    if rotation_first:
        return np.concatenate((vrot, vtrans))
    else:
        return np.concatenate((vtrans, vrot))
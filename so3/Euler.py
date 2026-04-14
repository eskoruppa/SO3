#!/bin/env python3

from __future__ import annotations

import numpy as np
import math
from ._pycondec import cond_jit
from .generators import _hat_map_sv

##########################################################################################################
############### SO3 Methods ##############################################################################
##########################################################################################################

# True zero-rotation cutoff (do NOT use for numerical stabilization)
DEF_EULER_EPSILON = 1e-12

# Thresholds for detecting angles close to 0 and close to pi via val = 0.5*(tr(R)-1)
DEF_EULER_CLOSE_TO_ONE = 1.0 - 1e-10
DEF_EULER_CLOSE_TO_MINUS_ONE = -1.0 + 1e-10

# Series / stability thresholds (kept outside for clarity + reproducibility)
DEF_EULER_SERIES_SMALL = 1e-4      # for sin(x)/x and (1-cos(x))/x^2
DEF_THETA_SCALE_SMALL = 1e-6       # for theta/(2 sin theta) scaling in log map
DEF_AXIS_NORM_EPS = 1e-15          # avoid division by ~0 when normalizing axis
DEF_AXIS_COMP_EPS = 1e-15          # avoid division by ~0 in pi-axis extraction


##########################################################################################################
# Internal single-vector JIT helpers
# Defined first so flat-batch helpers (below) can call them inside numba JIT context.
##########################################################################################################

# =========================
# so(3) → SO(3)
# =========================

@cond_jit(nopython=True, cache=True)
def _euler2rotmat_sv(Omega: np.ndarray) -> np.ndarray:
    Om = math.sqrt(Omega[0]*Omega[0] + Omega[1]*Omega[1] + Omega[2]*Omega[2])
    if Om < DEF_EULER_EPSILON:
        return np.eye(3, dtype=np.double)

    if Om < DEF_EULER_SERIES_SMALL:
        Om2 = Om * Om
        Om4 = Om2 * Om2
        A = 1.0 - Om2 / 6.0 + Om4 / 120.0
        B = 0.5 - Om2 / 24.0 + Om4 / 720.0
    else:
        A = math.sin(Om) / Om
        B = (1.0 - math.cos(Om)) / (Om * Om)

    x, y, z = Omega[0], Omega[1], Omega[2]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.empty((3, 3), dtype=np.double)
    R[0, 0] = 1.0 - B * (yy + zz)
    R[1, 1] = 1.0 - B * (xx + zz)
    R[2, 2] = 1.0 - B * (xx + yy)

    R[0, 1] = B * xy - A * z
    R[1, 0] = B * xy + A * z
    R[0, 2] = B * xz + A * y
    R[2, 0] = B * xz - A * y
    R[1, 2] = B * yz - A * x
    R[2, 1] = B * yz + A * x
    return R


# =========================
# SO(3) → so(3)
# =========================

@cond_jit(nopython=True, cache=True)
def _rotmat2euler_sv(R: np.ndarray) -> np.ndarray:
    out = np.empty(3, dtype=np.double)

    val = 0.5 * ((R[0, 0] + R[1, 1] + R[2, 2]) - 1.0)
    if val > 1.0:
        val = 1.0
    elif val < -1.0:
        val = -1.0

    # if val > DEF_EULER_CLOSE_TO_ONE:
    #     out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
    #     return out

    if val < DEF_EULER_CLOSE_TO_MINUS_ONE:
        r00, r11, r22 = R[0, 0], R[1, 1], R[2, 2]

        if (r00 >= r11) and (r00 >= r22):
            t = 0.5 * (r00 + 1.0);  t = 0.0 if t < 0.0 else t
            ax = math.sqrt(t)
            if ax < DEF_AXIS_COMP_EPS:
                out[0] = 0.0; out[1] = math.pi; out[2] = 0.0
                return out
            ay = R[0, 1] / (2.0 * ax)
            az = R[0, 2] / (2.0 * ax)
        elif r11 >= r22:
            t = 0.5 * (r11 + 1.0);  t = 0.0 if t < 0.0 else t
            ay = math.sqrt(t)
            if ay < DEF_AXIS_COMP_EPS:
                out[0] = math.pi; out[1] = 0.0; out[2] = 0.0
                return out
            ax = R[0, 1] / (2.0 * ay)
            az = R[1, 2] / (2.0 * ay)
        else:
            t = 0.5 * (r22 + 1.0);  t = 0.0 if t < 0.0 else t
            az = math.sqrt(t)
            if az < DEF_AXIS_COMP_EPS:
                out[0] = math.pi; out[1] = 0.0; out[2] = 0.0
                return out
            ax = R[0, 2] / (2.0 * az)
            ay = R[1, 2] / (2.0 * az)

        nrm = math.sqrt(ax*ax + ay*ay + az*az)
        if nrm < DEF_AXIS_NORM_EPS:
            out[0] = math.pi; out[1] = 0.0; out[2] = 0.0
            return out

        s = math.pi / nrm
        out[0] = ax * s; out[1] = ay * s; out[2] = az * s
        return out

    th = math.acos(val)
    vx = R[2, 1] - R[1, 2]
    vy = R[0, 2] - R[2, 0]
    vz = R[1, 0] - R[0, 1]

    if th < DEF_THETA_SCALE_SMALL:
        th2 = th * th
        scale = 0.5 + th2 / 12.0
    else:
        scale = 0.5 * th / math.sin(th)

    out[0] = scale * vx
    out[1] = scale * vy
    out[2] = scale * vz
    return out


@cond_jit(nopython=True, cache=True)
def _sqrt_rot_sv(R: np.ndarray) -> np.ndarray:
    """Rotation matrix for half the angle of R (same axis)."""
    return _euler2rotmat_sv(0.5 * _rotmat2euler_sv(R))


@cond_jit(nopython=True, cache=True)
def _midstep_sv(triad1: np.ndarray, triad2: np.ndarray) -> np.ndarray:
    return triad1 @ _sqrt_rot_sv(triad1.T @ triad2)


@cond_jit(nopython=True, cache=True)
def _right_jacobian_sv(Omega: np.ndarray) -> np.ndarray:
    Om = np.linalg.norm(Omega)
    Omega_hat = _hat_map_sv(Omega)
    Omega_hat_sq = Omega_hat @ Omega_hat
    if Om < 1e-6:
        return np.eye(3) - 0.5 * Omega_hat + (1.0 / 6.0) * Omega_hat_sq
    Om2 = Om * Om
    Om3 = Om2 * Om
    c1 = (1.0 - np.cos(Om)) / Om2
    c2 = (Om - np.sin(Om)) / Om3
    return np.eye(3) - c1 * Omega_hat + c2 * Omega_hat_sq


@cond_jit(nopython=True, cache=True)
def _left_jacobian_sv(Omega: np.ndarray) -> np.ndarray:
    return _right_jacobian_sv(Omega).T


@cond_jit(nopython=True, cache=True)
def _inverse_right_jacobian_sv(Omega: np.ndarray) -> np.ndarray:
    Om = np.linalg.norm(Omega)
    Omega_hat = _hat_map_sv(Omega)
    Omega_hat_sq = Omega_hat @ Omega_hat
    if Om < 1e-6:
        return np.eye(3) + 0.5 * Omega_hat + (1.0 / 12.0) * Omega_hat_sq
    Om2 = Om * Om
    # Stable formula: JR_inv = I + 1/2*hat + coeff*hat^2
    # coeff = 1/|Ω|² - cot(|Ω|/2)/(2|Ω|) = 1/|Ω|² - 1/(2|Ω|tan(|Ω|/2))
    # Singular only when tan(|Ω|/2) = 0, i.e. |Ω| = 2kπ (k≠0).
    tan_half = np.tan(Om / 2.0)
    if np.abs(tan_half) < 1e-10:
        # |Omega| is near 2k*pi (k≠0): J_R is singular at this angle.
        # The inverse right Jacobian is undefined; returning a matrix of infinities.
        return np.full((3, 3), np.inf)
    coeff = 1.0 / Om2 - 1.0 / (2.0 * Om * tan_half)
    return np.eye(3) + 0.5 * Omega_hat + coeff * Omega_hat_sq


@cond_jit(nopython=True, cache=True)
def _inverse_left_jacobian_sv(Omega: np.ndarray) -> np.ndarray:
    return _inverse_right_jacobian_sv(Omega).T


@cond_jit(nopython=True, cache=True)
def _se3_euler2rotmat_sv(Omega: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    if rotation_first:
        vrot = Omega[:3]
        vtrans = Omega[3:]
    else:
        vrot = Omega[3:]
        vtrans = Omega[:3]
    rotmat = np.zeros((4, 4))
    rotmat[:3, :3] = _euler2rotmat_sv(vrot)
    rotmat[:3, 3] = vtrans
    rotmat[3, 3] = 1
    return rotmat


@cond_jit(nopython=True, cache=True)
def _se3_rotmat2euler_sv(R: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    vrot = _rotmat2euler_sv(R[:3, :3])
    vtrans = R[:3, 3]
    if rotation_first:
        return np.concatenate((vrot, vtrans))
    else:
        return np.concatenate((vtrans, vrot))


##########################################################################################################
# Internal flat-batch JIT helpers
# All accept pre-flattened C-contiguous arrays and return flat (N, ...) output.
##########################################################################################################

@cond_jit(nopython=True, cache=True)
def _euler2rotmat_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3, 3))
    for i in range(n):
        result[i] = _euler2rotmat_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _rotmat2euler_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3))
    for i in range(n):
        result[i] = _rotmat2euler_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _sqrt_rot_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3, 3))
    for i in range(n):
        result[i] = _sqrt_rot_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _midstep_flat(t1_flat: np.ndarray, t2_flat: np.ndarray) -> np.ndarray:
    n = t1_flat.shape[0]
    result = np.zeros((n, 3, 3))
    for i in range(n):
        result[i] = _midstep_sv(t1_flat[i], t2_flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _right_jacobian_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3, 3))
    for i in range(n):
        result[i] = _right_jacobian_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _left_jacobian_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3, 3))
    for i in range(n):
        result[i] = _left_jacobian_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _inverse_right_jacobian_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3, 3))
    for i in range(n):
        result[i] = _inverse_right_jacobian_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _inverse_left_jacobian_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3, 3))
    for i in range(n):
        result[i] = _inverse_left_jacobian_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _se3_euler2rotmat_flat(flat: np.ndarray, rotation_first: bool) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 4, 4))
    for i in range(n):
        result[i] = _se3_euler2rotmat_sv(flat[i], rotation_first)
    return result


@cond_jit(nopython=True, cache=True)
def _se3_rotmat2euler_flat(flat: np.ndarray, rotation_first: bool) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 6))
    for i in range(n):
        result[i] = _se3_rotmat2euler_sv(flat[i], rotation_first)
    return result


##########################################################################################################
# Public batch functions
# Python dispatchers; shape-changing outputs prevent a single numba specialization.
# Detect single input by ndim: Euler vectors → ndim==1; rotation matrices → ndim==2.
##########################################################################################################

# =========================
# so(3) → SO(3)
# =========================

def euler2rotmat_batch(Omega: np.ndarray) -> np.ndarray:
    """Convert Euler (axis-angle) vector(s) to rotation matri(x/ces).

    Accepts any shape (..., 3). Output shape: (..., 3, 3).
    For a single vector (3,) returns a (3, 3) matrix.

    Args:
        Omega (np.ndarray): Euler vector(s), last dimension must be 3.

    Returns:
        np.ndarray: Rotation matrix/matrices of shape (..., 3, 3).
    """
    Omega = np.asarray(Omega, dtype=float)
    if Omega.ndim == 1:
        return _euler2rotmat_sv(Omega)
    orig_shape = Omega.shape
    n = Omega.size // 3
    flat = np.ascontiguousarray(Omega).reshape((n, 3))
    return _euler2rotmat_flat(flat).reshape(orig_shape[:-1] + (3, 3))


# =========================
# SO(3) → so(3)
# =========================

def rotmat2euler_batch(R: np.ndarray) -> np.ndarray:
    """Convert rotation matri(x/ces) to Euler (axis-angle) vector(s).

    A single matrix has shape (3, 3); a batch has shape (..., 3, 3).
    Output shape: (..., 3), or (3,) for a single matrix.

    Args:
        R (np.ndarray): Rotation matrix/matrices, last two dimensions must be (3, 3).

    Returns:
        np.ndarray: Euler vector(s) of shape (..., 3).
    """
    R = np.asarray(R, dtype=float)
    if R.ndim == 2:
        return _rotmat2euler_sv(R)
    orig_shape = R.shape          # (..., 3, 3)
    n = R.size // 9
    flat = np.ascontiguousarray(R).reshape((n, 3, 3))
    return _rotmat2euler_flat(flat).reshape(orig_shape[:-2] + (3,))


#########################################################################################################
############## sqrt of rotation matrix ##################################################################
#########################################################################################################

def sqrt_rot_batch(R: np.ndarray) -> np.ndarray:
    """Rotation matrix for the same axis but half the angle.

    A single matrix has shape (3, 3); a batch has shape (..., 3, 3).
    Output shape matches input shape.

    Args:
        R (np.ndarray): Rotation matrix/matrices, last two dimensions must be (3, 3).

    Returns:
        np.ndarray: Square-root rotation matrix/matrices of shape (..., 3, 3).
    """
    R = np.asarray(R, dtype=float)
    if R.ndim == 2:
        return _sqrt_rot_sv(R)
    orig_shape = R.shape
    n = R.size // 9
    flat = np.ascontiguousarray(R).reshape((n, 3, 3))
    return _sqrt_rot_flat(flat).reshape(orig_shape)


def midstep_batch(triad1: np.ndarray, triad2: np.ndarray) -> np.ndarray:
    """Mid-step frame between two triads (rotation matrices).

    Both inputs must have the same shape. A single pair has shape (3, 3);
    a batch has shape (..., 3, 3). Output shape matches input shape.

    Args:
        triad1 (np.ndarray): First rotation matrix/matrices, shape (..., 3, 3).
        triad2 (np.ndarray): Second rotation matrix/matrices, shape (..., 3, 3).

    Returns:
        np.ndarray: Mid-step rotation matrix/matrices of shape (..., 3, 3).
    """
    triad1 = np.asarray(triad1, dtype=float)
    triad2 = np.asarray(triad2, dtype=float)
    if triad1.ndim == 2:
        return _midstep_sv(triad1, triad2)
    orig_shape = triad1.shape
    n = triad1.size // 9
    t1_flat = np.ascontiguousarray(triad1).reshape((n, 3, 3))
    t2_flat = np.ascontiguousarray(triad2).reshape((n, 3, 3))
    return _midstep_flat(t1_flat, t2_flat).reshape(orig_shape)


##########################################################################################################
############### Left- and Right-Jacobians ################################################################
##########################################################################################################

def right_jacobian_batch(Omega: np.ndarray) -> np.ndarray:
    """Compute the right Jacobian of SO(3) exponential map.

    The right Jacobian encodes how small perturbations in the Lie algebra
    map to perturbations in the group via the exponential map:
        exp(Omega + δΩ) ≈ exp(Omega) exp(hat[JR(Omega) δΩ])

    Accepts any shape (..., 3). Output shape: (..., 3, 3).

    Parameters
    ----------
    Omega : ndarray, shape (..., 3)
        Rotation vector(s) (axis-angle, radians).

    Returns
    -------
    JR : ndarray, shape (..., 3, 3)
        Right Jacobian matrix/matrices.
    """
    Omega = np.asarray(Omega, dtype=float)
    if Omega.ndim == 1:
        return _right_jacobian_sv(Omega)
    orig_shape = Omega.shape
    n = Omega.size // 3
    flat = np.ascontiguousarray(Omega).reshape((n, 3))
    return _right_jacobian_flat(flat).reshape(orig_shape[:-1] + (3, 3))


def left_jacobian_batch(Omega: np.ndarray) -> np.ndarray:
    """Compute the left Jacobian of SO(3) exponential map.

    The left Jacobian encodes how small perturbations in the Lie algebra
    map to perturbations in the group via the exponential map:
        exp(Omega + δΩ) ≈ exp(hat[JL(Omega) δΩ]) exp(Omega)

    Accepts any shape (..., 3). Output shape: (..., 3, 3).

    Parameters
    ----------
    Omega : ndarray, shape (..., 3)
        Rotation vector(s) (axis-angle, radians).

    Returns
    -------
    JL : ndarray, shape (..., 3, 3)
        Left Jacobian matrix/matrices.
    """
    Omega = np.asarray(Omega, dtype=float)
    if Omega.ndim == 1:
        return _left_jacobian_sv(Omega)
    orig_shape = Omega.shape
    n = Omega.size // 3
    flat = np.ascontiguousarray(Omega).reshape((n, 3))
    return _left_jacobian_flat(flat).reshape(orig_shape[:-1] + (3, 3))


def inverse_right_jacobian_batch(Omega: np.ndarray) -> np.ndarray:
    """Compute the inverse of the right Jacobian.

    Accepts any shape (..., 3). Output shape: (..., 3, 3).

    Parameters
    ----------
    Omega : ndarray, shape (..., 3)
        Rotation vector(s) (axis-angle, radians).

    Returns
    -------
    JR_inv : ndarray, shape (..., 3, 3)
        Inverse of the right Jacobian matrix/matrices.
    """
    Omega = np.asarray(Omega, dtype=float)
    if Omega.ndim == 1:
        return _inverse_right_jacobian_sv(Omega)
    orig_shape = Omega.shape
    n = Omega.size // 3
    flat = np.ascontiguousarray(Omega).reshape((n, 3))
    return _inverse_right_jacobian_flat(flat).reshape(orig_shape[:-1] + (3, 3))


def inverse_left_jacobian_batch(Omega: np.ndarray) -> np.ndarray:
    """Compute the inverse of the left Jacobian.

    Accepts any shape (..., 3). Output shape: (..., 3, 3).

    Parameters
    ----------
    Omega : ndarray, shape (..., 3)
        Rotation vector(s) (axis-angle, radians).

    Returns
    -------
    JL_inv : ndarray, shape (..., 3, 3)
        Inverse of the left Jacobian matrix/matrices.
    """
    Omega = np.asarray(Omega, dtype=float)
    if Omega.ndim == 1:
        return _inverse_left_jacobian_sv(Omega)
    orig_shape = Omega.shape
    n = Omega.size // 3
    flat = np.ascontiguousarray(Omega).reshape((n, 3))
    return _inverse_left_jacobian_flat(flat).reshape(orig_shape[:-1] + (3, 3))


##########################################################################################################
############### SE3 Methods ##############################################################################
##########################################################################################################

def se3_euler2rotmat_batch(Omega: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Convert SE(3) Euler 6-vector(s) to 4x4 SE(3) matri(x/ces).

    A single vector has shape (6,); a batch has shape (..., 6).
    Output shape: (4, 4) for single, (..., 4, 4) for batch.

    Args:
        Omega (np.ndarray): SE(3) coordinate vector(s), last dimension must be 6.
        rotation_first (bool): If True, first 3 components are rotation (default True).

    Returns:
        np.ndarray: SE(3) matrix/matrices of shape (..., 4, 4).
    """
    Omega = np.asarray(Omega, dtype=float)
    if Omega.ndim == 1:
        return _se3_euler2rotmat_sv(Omega, rotation_first)
    orig_shape = Omega.shape
    n = Omega.size // 6
    flat = np.ascontiguousarray(Omega).reshape((n, 6))
    return _se3_euler2rotmat_flat(flat, rotation_first).reshape(orig_shape[:-1] + (4, 4))


def se3_rotmat2euler_batch(R: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Convert 4x4 SE(3) matri(x/ces) to SE(3) Euler 6-vector(s).

    A single matrix has shape (4, 4); a batch has shape (..., 4, 4).
    Output shape: (6,) for single, (..., 6) for batch.

    Args:
        R (np.ndarray): SE(3) matrix/matrices, last two dimensions must be (4, 4).
        rotation_first (bool): If True, first 3 components are rotation (default True).

    Returns:
        np.ndarray: SE(3) coordinate vector(s) of shape (..., 6).
    """
    R = np.asarray(R, dtype=float)
    if R.ndim == 2:
        return _se3_rotmat2euler_sv(R, rotation_first)
    orig_shape = R.shape
    n = R.size // 16
    flat = np.ascontiguousarray(R).reshape((n, 4, 4))
    return _se3_rotmat2euler_flat(flat, rotation_first).reshape(orig_shape[:-2] + (6,))


def se3_eulers2rotmats(X: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Wrapper around se3_euler2rotmat (retained for API compatibility).

    Convert 6D SE(3) coordinates to SE(3) matrices.
    Accepts shape (6,) or (..., 6). Returns (4, 4) or (..., 4, 4).
    """
    return se3_euler2rotmat_batch(X, rotation_first=rotation_first)


def se3_rotmats2eulers(se3: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    """Wrapper around se3_rotmat2euler (retained for API compatibility).

    Convert SE(3) matrices to 6D coordinates.
    Accepts shape (4, 4) or (..., 4, 4). Returns (6,) or (..., 6).
    """
    return se3_rotmat2euler_batch(se3, rotation_first=rotation_first)


##########################################################################################################
# Public single-vector API
# Direct aliases to the internal JIT _sv helpers.
# All _single functions accept a single vector/matrix and are JIT-callable.
##########################################################################################################

# #: Single-vector version of :func:`euler2rotmat`. Input (3,) → output (3, 3). JIT-callable.
# euler2rotmat_single = _euler2rotmat_sv

# #: Single-matrix version of :func:`rotmat2euler`. Input (3, 3) → output (3,). JIT-callable.
# rotmat2euler_single = _rotmat2euler_sv

# #: Single-matrix version of :func:`sqrt_rot`. Input (3, 3) → output (3, 3). JIT-callable.
# sqrt_rot_single = _sqrt_rot_sv

# #: Single-pair version of :func:`midstep`. Inputs (3, 3), (3, 3) → output (3, 3). JIT-callable.
# midstep_single = _midstep_sv

# #: Single-vector version of :func:`right_jacobian`. Input (3,) → output (3, 3). JIT-callable.
# right_jacobian_single = _right_jacobian_sv

# #: Single-vector version of :func:`left_jacobian`. Input (3,) → output (3, 3). JIT-callable.
# left_jacobian_single = _left_jacobian_sv

# #: Single-vector version of :func:`inverse_right_jacobian`. Input (3,) → output (3, 3). JIT-callable.
# inverse_right_jacobian_single = _inverse_right_jacobian_sv

# #: Single-vector version of :func:`inverse_left_jacobian`. Input (3,) → output (3, 3). JIT-callable.
# inverse_left_jacobian_single = _inverse_left_jacobian_sv

# #: Single-vector version of :func:`se3_euler2rotmat`. Input (6,) → output (4, 4). JIT-callable.
# se3_euler2rotmat_single = _se3_euler2rotmat_sv

# #: Single-matrix version of :func:`se3_rotmat2euler`. Input (4, 4) → output (6,). JIT-callable.
# se3_rotmat2euler_single = _se3_rotmat2euler_sv


# =========================
# so(3) → SO(3)
# =========================

@cond_jit(nopython=True, cache=True)
def euler2rotmat_single(Omega: np.ndarray) -> np.ndarray:
    Om = math.sqrt(Omega[0]*Omega[0] + Omega[1]*Omega[1] + Omega[2]*Omega[2])
    if Om < DEF_EULER_EPSILON:
        return np.eye(3, dtype=np.double)

    if Om < DEF_EULER_SERIES_SMALL:
        Om2 = Om * Om
        Om4 = Om2 * Om2
        A = 1.0 - Om2 / 6.0 + Om4 / 120.0
        B = 0.5 - Om2 / 24.0 + Om4 / 720.0
    else:
        A = math.sin(Om) / Om
        B = (1.0 - math.cos(Om)) / (Om * Om)

    x, y, z = Omega[0], Omega[1], Omega[2]
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z

    R = np.empty((3, 3), dtype=np.double)
    R[0, 0] = 1.0 - B * (yy + zz)
    R[1, 1] = 1.0 - B * (xx + zz)
    R[2, 2] = 1.0 - B * (xx + yy)

    R[0, 1] = B * xy - A * z
    R[1, 0] = B * xy + A * z
    R[0, 2] = B * xz + A * y
    R[2, 0] = B * xz - A * y
    R[1, 2] = B * yz - A * x
    R[2, 1] = B * yz + A * x
    return R


# =========================
# SO(3) → so(3)
# =========================

@cond_jit(nopython=True, cache=True)
def rotmat2euler_single(R: np.ndarray) -> np.ndarray:
    out = np.empty(3, dtype=np.double)

    val = 0.5 * ((R[0, 0] + R[1, 1] + R[2, 2]) - 1.0)
    if val > 1.0:
        val = 1.0
    elif val < -1.0:
        val = -1.0

    # if val > DEF_EULER_CLOSE_TO_ONE:
    #     out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
    #     return out

    if val < DEF_EULER_CLOSE_TO_MINUS_ONE:
        r00, r11, r22 = R[0, 0], R[1, 1], R[2, 2]

        # pick dominant diagonal; compute axis; deterministic fallback if denom tiny
        if (r00 >= r11) and (r00 >= r22):
            t = 0.5 * (r00 + 1.0);  t = 0.0 if t < 0.0 else t
            ax = math.sqrt(t)
            if ax < DEF_AXIS_COMP_EPS:
                out[0] = 0.0; out[1] = math.pi; out[2] = 0.0
                return out
            ay = R[0, 1] / (2.0 * ax)
            az = R[0, 2] / (2.0 * ax)
        elif r11 >= r22:
            t = 0.5 * (r11 + 1.0);  t = 0.0 if t < 0.0 else t
            ay = math.sqrt(t)
            if ay < DEF_AXIS_COMP_EPS:
                out[0] = math.pi; out[1] = 0.0; out[2] = 0.0
                return out
            ax = R[0, 1] / (2.0 * ay)
            az = R[1, 2] / (2.0 * ay)
        else:
            t = 0.5 * (r22 + 1.0);  t = 0.0 if t < 0.0 else t
            az = math.sqrt(t)
            if az < DEF_AXIS_COMP_EPS:
                out[0] = math.pi; out[1] = 0.0; out[2] = 0.0
                return out
            ax = R[0, 2] / (2.0 * az)
            ay = R[1, 2] / (2.0 * az)

        nrm = math.sqrt(ax*ax + ay*ay + az*az)
        if nrm < DEF_AXIS_NORM_EPS:
            out[0] = math.pi; out[1] = 0.0; out[2] = 0.0
            return out

        s = math.pi / nrm
        out[0] = ax * s; out[1] = ay * s; out[2] = az * s
        return out

    th = math.acos(val)
    vx = R[2, 1] - R[1, 2]
    vy = R[0, 2] - R[2, 0]
    vz = R[1, 0] - R[0, 1]

    if th < DEF_THETA_SCALE_SMALL:
        th2 = th * th
        scale = 0.5 + th2 / 12.0
    else:
        scale = 0.5 * th / math.sin(th)

    out[0] = scale * vx
    out[1] = scale * vy
    out[2] = scale * vz
    return out


#########################################################################################################
############## sqrt of rotation matrix ##################################################################
#########################################################################################################


@cond_jit(nopython=True,cache=True)
def sqrt_rot_single(R: np.ndarray) -> np.ndarray:
    """generates rotation matrix that corresponds to a rotation over the same axis, but over half the angle."""
    return euler2rotmat_single(0.5 * rotmat2euler_single(R))


@cond_jit(nopython=True,cache=True)
def midstep_single(triad1: np.ndarray, triad2: np.ndarray) -> np.ndarray:
    return triad1 @ sqrt_rot_single(triad1.T @ triad2)


##########################################################################################################
############### Left- and Right-Jacobians ################################################################
##########################################################################################################

@cond_jit(nopython=True,cache=True)
def right_jacobian_single(Omega: np.ndarray) -> np.ndarray:
    """Compute the right Jacobian of SO(3) exponential map.

    The right Jacobian encodes how small perturbations in the Lie algebra
    map to perturbations in the group via the exponential map:
        exp(Omega + δΩ) ≈ exp(Omega) exp(hat[JR(Omega) δΩ])

    For small Ω use Taylor expansion (not zeroth order).

    Parameters
    ----------
    Omega : ndarray, shape (3,)
        Rotation vector (axis-angle, radians).

    Returns
    -------
    JR : ndarray, shape (3, 3)
        Right Jacobian matrix.

    References
    ----------
    Equation (A14) in NucFreeEnergy.pdf:
        JR(Ω) = 1 - (1-cos Ω)/Ω² * Ω̂ + (Ω - sin Ω)/Ω³ * Ω̂²
    """
    Om = np.linalg.norm(Omega)
    Omega_hat = _hat_map_sv(Omega)
    Omega_hat_sq = Omega_hat @ Omega_hat

    if Om < 1e-6:
        # Small-angle expansion: JR(Ω) ≈ I - (1/2)Ω̂ + (1/6)Ω̂²  [right_jacobian_single]
        return np.eye(3) - 0.5 * Omega_hat + (1.0 / 6.0) * Omega_hat_sq

    Om2 = Om * Om
    Om3 = Om2 * Om

    c1 = (1.0 - np.cos(Om)) / Om2
    c2 = (Om - np.sin(Om)) / Om3

    return np.eye(3) - c1 * Omega_hat + c2 * Omega_hat_sq


@cond_jit(nopython=True,cache=True)
def left_jacobian_single(Omega: np.ndarray) -> np.ndarray:
    """Compute the left Jacobian of SO(3) exponential map.

    The left Jacobian encodes how small perturbations in the Lie algebra
    map to perturbations in the group via the exponential map:
        exp(Omega + δΩ) ≈ exp(hat[JL(Omega) δΩ]) exp(Omega)

    For small Ω use Taylor expansion (not zeroth order).

    Parameters
    ----------
    Omega : ndarray, shape (3,)
        Rotation vector (axis-angle, radians).

    Returns
    -------
    JL : ndarray, shape (3, 3)
        Left Jacobian matrix.

    References
    ----------
    Equation (A15) in NucFreeEnergy.pdf:
        JL(Ω) = JR(-Ω) = JR^T(Ω) = R(Ω) JR(Ω)
    where R(Ω) = exp(Ω̂) is the rotation matrix.
    """
    # Use the relation: JL(Ω) = JR^T(Ω)
    return right_jacobian_single(Omega).T


@cond_jit(nopython=True,cache=True)
def inverse_right_jacobian_single(Omega: np.ndarray) -> np.ndarray:
    """Compute the inverse of the right Jacobian.

    Parameters
    ----------
    Omega : ndarray, shape (3,)
        Rotation vector (axis-angle, radians). Must have |Omega| < 2π;
        the right Jacobian is singular at |Omega| = 2kπ (k≠0).

    Returns
    -------
    JR_inv : ndarray, shape (3, 3)
        Inverse of the right Jacobian matrix.

    Notes
    -----
    For small Ω, the inverse is computed via Taylor expansion:
        JR_inv(Ω) ≈ I + (1/2)Ω̂ + (1/12)Ω̂²

    For general angles, the exact formula is:
        JR_inv(Ω) = I + (1/2)Ω̂ + coeff(Ω) * Ω̂²
    where coeff(Ω) = 1/|Ω|² - cot(|Ω|/2)/(2|Ω|)
                   = 1/|Ω|² - 1/(2|Ω|tan(|Ω|/2))
    This is singular only at |Omega| = 2kπ (k≠0) where tan(|Ω|/2) = 0
    and J_R is singular (eigenvalue zero in the plane perpendicular to Ω).
    """
    Om = np.linalg.norm(Omega)
    Omega_hat = _hat_map_sv(Omega)
    Omega_hat_sq = Omega_hat @ Omega_hat

    if Om < 1e-6:
        # Small-angle expansion: JR_inv(Ω) ≈ I + (1/2)Ω̂ + (1/12)Ω̂²
        return np.eye(3) + 0.5 * Omega_hat + (1.0 / 12.0) * Omega_hat_sq

    Om2 = Om * Om
    # Stable formula: JR_inv = I + 1/2*hat + coeff*hat^2
    # coeff = 1/|Ω|² - cot(|Ω|/2)/(2|Ω|) = 1/|Ω|² - 1/(2|Ω|tan(|Ω|/2))
    # Singular only when tan(|Ω|/2) = 0, i.e. |Ω| = 2kπ (k≠0).
    tan_half = np.tan(Om / 2.0)
    if np.abs(tan_half) < 1e-10:
        # |Omega| ≈ 2kπ (k≠0): J_R is singular, inverse is undefined.
        return np.full((3, 3), np.inf)
    coeff = 1.0 / Om2 - 1.0 / (2.0 * Om * tan_half)

    return np.eye(3) + 0.5 * Omega_hat + coeff * Omega_hat_sq


@cond_jit(nopython=True,cache=True)
def inverse_left_jacobian_single(Omega: np.ndarray) -> np.ndarray:
    """Compute the inverse of the left Jacobian.

    Parameters
    ----------
    Omega : ndarray, shape (3,)
        Rotation vector (axis-angle, radians).

    Returns
    -------
    JL_inv : ndarray, shape (3, 3)
        Inverse of the left Jacobian matrix.

    Notes
    -----
    Uses the relation: JL_inv(Ω) = (JR_inv(Ω))^T
    """
    return inverse_right_jacobian_single(Omega).T


##########################################################################################################
############### SE3 Methods ##############################################################################
##########################################################################################################


@cond_jit(nopython=True,cache=True)
def se3_euler2rotmat_single(Omega: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    # if Omega.shape != (6,):
    #     raise ValueError(f'Expected shape (6,) array, but encountered {Omega.shape}.')
    if rotation_first:
        vrot = Omega[:3]
        vtrans = Omega[3:]
    else:
        vrot = Omega[3:]
        vtrans = Omega[:3]
    rotmat = np.zeros((4, 4))
    rotmat[:3, :3] = euler2rotmat_single(vrot)
    rotmat[:3, 3] = vtrans
    rotmat[3, 3] = 1
    return rotmat


@cond_jit(nopython=True,cache=True)
def se3_rotmat2euler_single(R: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    # if R.shape != (4,4):
    #     raise ValueError(f'Expected shape (4,4) array, but encountered {R.shape}.')
    vrot = rotmat2euler_single(R[:3, :3])
    vtrans = R[:3, 3]
    if rotation_first:
        return np.concatenate((vrot, vtrans))
    else:
        return np.concatenate((vtrans, vrot))
    
    
def se3_eulers2rotmats_single(X: np.ndarray) -> np.ndarray:
    """
    Convert 6D SE(3) coordinates to SE(3) matrices via se3_euler2rotmat.

    Accepts:
      - a single coordinate vector: shape (6,)
      - a batch: shape (..., 6)

    Returns:
      - a single SE(3) matrix for (6,) input
      - a batch of SE(3) matrices for (..., 6) input, with shape (..., M, M)
        where M is whatever se3_euler2rotmat returns (typically 4x4 or 3x4).
    """
    X = np.asarray(X)
    
    if X.ndim == 1:
        if X.shape[0] != 6:
            raise ValueError(f"Expected shape (6,), got {X.shape}.")
        return se3_euler2rotmat_single(X)

    if X.shape[-1] != 6:
        raise ValueError(f"Expected last dimension to be 6, got {X.shape}.")

    lead_shape = X.shape[:-1]
    n = int(np.prod(lead_shape))
    X_flat = X.reshape((n, 6))

    try:
        se3_flat = se3_euler2rotmat_single(X_flat)
        se3_flat = np.asarray(se3_flat)
        return se3_flat.reshape(lead_shape + se3_flat.shape[-2:])
    except Exception:
        first = np.asarray(se3_euler2rotmat_single(X_flat[0]))
        out = np.empty((n,) + first.shape, dtype=first.dtype)
        out[0] = first
        for i in range(1, n):
            out[i] = se3_euler2rotmat_single(X_flat[i])
        return out.reshape(lead_shape + first.shape)

    
def se3_rotmats2eulers_single(se3: np.ndarray) -> np.ndarray:
    """
    Convert SE(3) matrices to 6D coordinates via se3_rotmat2euler.

    Accepts:
      - a single matrix: shape (M, M) (typically (4,4) or (3,4))
      - a batch: shape (..., M, M)

    Returns:
      - shape (6,) for a single matrix
      - shape (..., 6) for a batch
    """
    se3 = np.asarray(se3)

    if se3.ndim == 2:
        return se3_rotmat2euler_single(se3)

    lead_shape = se3.shape[:-2]
    n = int(np.prod(lead_shape))
    se3_flat = se3.reshape((n,) + se3.shape[-2:])

    try:
        X_flat = se3_rotmat2euler_single(se3_flat)
        X_flat = np.asarray(X_flat, dtype=np.float64)
        return X_flat.reshape(lead_shape + (6,))
    except Exception:
        X_flat = np.empty((n, 6), dtype=np.float64)
        for i in range(n):
            X_flat[i] = se3_rotmat2euler_single(se3_flat[i])
        return X_flat.reshape(lead_shape + (6,))

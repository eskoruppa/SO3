"""Utilities to convert between 6D SE(3) coordinates and homogeneous transforms.

Conventions
-----------
- 6D vectors `X` are laid out as `[omega, v]` where `omega` is a rotation vector
    (axis-angle, radians) and `v` is a translation vector.
- Rotation-vector ↔ rotation-matrix conversions use `euler2rotmat` /
    `rotmat2euler` (the package's rotation-vector / Rodrigues convention).

Functions implement both direct SE(3) conversions and left/right-midpoint
representations (`glh`, `grh`) such that `g = glh @ grh`.
"""
from __future__ import annotations

import numpy as np
from ._pycondec import cond_jit
from .Euler import _euler2rotmat_sv as euler2rotmat
from .Euler import _rotmat2euler_sv as rotmat2euler
from .Euler import _right_jacobian_sv as right_jacobian
from .Euler import _inverse_right_jacobian_sv as inverse_right_jacobian
from .generators import _hat_map_sv as hat_map

@cond_jit(nopython=True,cache=True)
def X_inv(X: np.ndarray) -> np.ndarray:
    """Compute the inverse of a 6D SE(3) coordinate vector.

    Parameters
    ----------
    X : ndarray, shape (6,)
        SE(3) coordinate vector `[omega, v]`.

    Returns
    -------
    X_inv : ndarray, shape (6,)
        Inverse SE(3) coordinate vector corresponding to `X`.
    """
    Xinv = np.zeros((6,), dtype=np.float64)
    Rinv = euler2rotmat(-X[:3])
    Xinv[:3] = -X[:3]
    Xinv[3:] = -Rinv @ X[3:]
    return Xinv

@cond_jit(nopython=True,cache=True)
def X2g(X: np.ndarray) -> np.ndarray:
    """Convert a 6D SE(3) coordinate vector to a 4x4 homogeneous matrix.

    Parameters
    ----------
    X : ndarray, shape (6,)
        6D vector `[omega, v]` where `omega` is the rotation vector
        (axis-angle, radians) and `v` is the translation.

    Returns
    -------
    g : ndarray, shape (4, 4)
        Homogeneous transformation matrix in SE(3). Rotation is placed in
        the top-left 3x3 block and translation in the top-right 3x1 column.
    """
    g = np.zeros((4, 4), dtype=np.float64)
    R = euler2rotmat(X[:3])
    w = X[3:]
    g[:3, :3] = R
    g[:3, 3] = w
    g[3, 3] = 1.0
    return g

@cond_jit(nopython=True,cache=True)
def g2X(g: np.ndarray) -> np.ndarray:
    """Convert a 4x4 homogeneous transform to a 6D coordinate vector.

    Parameters
    ----------
    g : ndarray, shape (4, 4)
        Homogeneous transformation matrix.

    Returns
    -------
    X : ndarray, shape (6,)
        6D vector `[omega, v]` where `omega` is the rotation vector
        (axis-angle, radians) recovered from the rotation block and `v` is
        the translation (top-right column).
    """
    X = np.zeros((6,), dtype=np.float64)
    R = g[:3, :3]
    w = g[:3, 3]
    X[:3] = rotmat2euler(R)
    X[3:] = w
    return X

@cond_jit(nopython=True,cache=True)
def X2glh(X: np.ndarray) -> np.ndarray:
    """Compute the left-midpoint homogeneous transform for a 6D vector.

    This returns `g_lh = exp(0.5 * xi)` in matrix form where `xi = [omega, v]`.
    For this left-midpoint representation the translation in `g_lh` is simply
    half the original translation.

    Parameters
    ----------
    X : ndarray, shape (6,)
        SE(3) coordinate vector `[omega, v]`.

    Returns
    -------
    glh : ndarray, shape (4,4)
        Left-midpoint homogeneous transform.
    """
    glh = np.zeros((4, 4), dtype=np.float64)
    glh[:3, :3] = euler2rotmat(0.5 * X[:3])
    glh[:3, 3] = 0.5 * X[3:]
    glh[3, 3] = 1.0
    return glh

@cond_jit(nopython=True,cache=True)
def X2grh(X: np.ndarray) -> np.ndarray:
    """Compute the right-midpoint homogeneous transform for a 6D vector.

    The right-midpoint `g_rh` is chosen so that the full transform decomposes as
    `g = g_lh @ g_rh`. The rotation part is the half-rotation and the
    translation is expressed in the right (local) frame:
    `t_rh = 0.5 * R_half.T @ v`.

    Parameters
    ----------
    X : ndarray, shape (6,)
        SE(3) coordinate vector `[omega, v]`.

    Returns
    -------
    grh : ndarray, shape (4,4)
        Right-midpoint homogeneous transform.
    """
    grh = np.zeros((4, 4), dtype=np.float64)
    sqrtR = euler2rotmat(0.5 * X[:3])
    grh[:3, :3] = sqrtR
    grh[:3, 3] = 0.5 * sqrtR.T @ X[3:]
    grh[3, 3] = 1.0
    return grh

@cond_jit(nopython=True,cache=True)
def glh2X(glh: np.ndarray) -> np.ndarray:
    """Recover the 6D SE(3) vector from a left-midpoint homogeneous matrix.

    Parameters
    ----------
    glh : ndarray, shape (4,4)
        Left-midpoint homogeneous transform (g_lh = exp(0.5*xi)).

    Returns
    -------
    X : ndarray, shape (6,)
        Reconstructed SE(3) vector `[omega, v]`.
    """
    X = np.zeros((6,), dtype=np.float64)
    R_lh = glh[:3, :3]
    w_lh = glh[:3, 3]
    X[:3] = 2 * rotmat2euler(R_lh)
    X[3:] = 2 * w_lh
    return X

@cond_jit(nopython=True,cache=True)
def grh2X(grh: np.ndarray) -> np.ndarray:
    """Recover the 6D SE(3) vector from a right-midpoint homogeneous matrix.

    Parameters
    ----------
    grh : ndarray, shape (4,4)
        Right-midpoint homogeneous transform.

    Returns
    -------
    X : ndarray, shape (6,)
        Reconstructed SE(3) vector `[omega, v]`.

    Notes
    -----
    Because the right-midpoint stores the translation in the local frame,
    the global translation is recovered as `v = 2 * R_rh @ t_rh`.
    """
    X = np.zeros((6,), dtype=np.float64)
    R_rh = grh[:3, :3]
    w_rh = grh[:3, 3]
    X[:3] = 2 * rotmat2euler(R_rh)
    X[3:] = 2 * R_rh @ w_rh
    return X
    
@cond_jit(nopython=True,cache=True)
def g2glh(g: np.ndarray) -> np.ndarray:
    """Compute left-midpoint `glh` directly from homogeneous matrix `g`.

    This is a thin wrapper: `g2glh(g) == X2glh(g2X(g))`.
    """
    return X2glh(g2X(g))

@cond_jit(nopython=True,cache=True)
def g2grh(g: np.ndarray) -> np.ndarray:
    """Compute right-midpoint `grh` directly from homogeneous matrix `g`.

    This is a thin wrapper: `g2grh(g) == X2grh(g2X(g))`.
    """
    return X2grh(g2X(g))

@cond_jit(nopython=True,cache=True)
def glh2g(glh: np.ndarray) -> np.ndarray:
    """Reconstruct full homogeneous matrix `g` from left-midpoint `glh`.

    This is a thin wrapper around `X2g(glh2X(glh))`.
    """
    return X2g(glh2X(glh))

@cond_jit(nopython=True,cache=True)
def grh2g(grh: np.ndarray) -> np.ndarray:
    """Reconstruct full homogeneous matrix `g` from right-midpoint `grh`.

    This is a thin wrapper around `X2g(grh2X(grh))`.
    """
    return X2g(grh2X(grh))

@cond_jit(nopython=True,cache=True)
def g2glh_inv(g: np.ndarray) -> np.ndarray:
    """Compute left-midpoint `glh` directly from homogeneous matrix `g`.

    This is a thin wrapper: `g2glh(g) == X2glh(g2X(g))`.
    """
    return X2glh_inv(g2X(g))

@cond_jit(nopython=True,cache=True)
def g2grh_inv(g: np.ndarray) -> np.ndarray:
    """Compute right-midpoint `grh` directly from homogeneous matrix `g`.

    This is a thin wrapper: `g2grh(g) == X2grh(g2X(g))`.
    """
    return X2grh_inv(g2X(g))

@cond_jit(nopython=True,cache=True)
def glh2g_inv(glh: np.ndarray) -> np.ndarray:
    """Reconstruct full homogeneous matrix `g` from left-midpoint `glh`.

    This is a thin wrapper around `X2g(glh2X(glh))`.
    """
    return X2g_inv(glh2X(glh))

@cond_jit(nopython=True,cache=True)
def grh2g_inv(grh: np.ndarray) -> np.ndarray:
    """Reconstruct full homogeneous matrix `g` from right-midpoint `grh`.

    This is a thin wrapper around `X2g(grh2X(grh))`.
    """
    return X2g_inv(grh2X(grh))

@cond_jit(nopython=True,cache=True)
def X2g_inv(X: np.ndarray) -> np.ndarray:
    g = np.zeros((4, 4), dtype=np.float64)
    R = euler2rotmat(X[:3])
    w = X[3:]
    g[:3, :3] = R.T
    g[:3, 3] = -R.T @ w
    g[3, 3] = 1.0
    return g

@cond_jit(nopython=True,cache=True)
def X2glh_inv(X: np.ndarray) -> np.ndarray:
    glh = np.zeros((4, 4), dtype=np.float64)
    sqrtR = euler2rotmat(0.5 * X[:3])
    glh[:3, :3] = sqrtR.T
    glh[:3, 3] = -0.5 * sqrtR.T @ X[3:]
    glh[3, 3] = 1.0
    return glh

@cond_jit(nopython=True,cache=True)
def X2grh_inv(X: np.ndarray) -> np.ndarray:
    grh = np.zeros((4, 4), dtype=np.float64)
    sqrtR = euler2rotmat(0.5 * X[:3])
    grh[:3, :3] = sqrtR.T
    grh[:3, 3] = -0.5 * sqrtR.T @ sqrtR.T @ X[3:]
    grh[3, 3] = 1.0
    return grh

@cond_jit(nopython=True,cache=True)
def A_rev():
    return -np.eye(6)

@cond_jit(nopython=True,cache=True)
def A_lh(X0: np.ndarray) -> np.ndarray:
    """Compute the transformation matrix that converts dynamic junction coordinates corresponding to full junctions to dynamic left-hand half step junction coordinates.

    Parameters
    ----------
    X0 : ndarray, shape (6,)
        SE(3) coordinate vector at the reference point.

    Returns
    -------
    A : ndarray, shape (6, 6)
        Transformation matrix linearized around X0.
    """
    A = np.zeros((6, 6), dtype=np.float64)

    Jr_phihalf = right_jacobian(0.5 * X0[:3])
    Jri_phi    = inverse_right_jacobian(X0[:3])
    
    A[:3,:3] = Jr_phihalf @ Jri_phi
    A[3:,3:] = euler2rotmat(0.5*X0[:3])
    return 0.5*A

@cond_jit(nopython=True,cache=True)
def A_rh(X0: np.ndarray) -> np.ndarray:
    """Compute the transformation matrix that converts dynamic junction coordinates corresponding to full junctions to dynamic right-hand half step junction coordinates.

    Parameters
    ----------
    X0 : ndarray, shape (6,)
        SE(3) coordinate vector at the reference point.

    Returns
    -------
    A : ndarray, shape (6, 6)
        Transformation matrix linearized around X0.
    """
    A = np.zeros((6, 6), dtype=np.float64)

    Jr_phihalf = right_jacobian(0.5 * X0[:3])
    Jri_phi    = inverse_right_jacobian(X0[:3])
    jac_prod = Jr_phihalf @ Jri_phi
    sqrtS = euler2rotmat(0.5*X0[:3])
    S = sqrtS @ sqrtS
    shat = hat_map(X0[3:])
 
    A[:3,:3] = jac_prod
    A[3:,3:] = np.eye(3)
    A[3:,:3] = 0.5 * S.T @ shat @ sqrtS @ jac_prod
    return 0.5*A    

#!/bin/env python3

from __future__ import annotations

import numpy as np
import math
from .pyConDec.pycondec import cond_jit
from .generators import hat_map

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


# =========================
# so(3) → SO(3)
# =========================

@cond_jit(nopython=True, cache=True)
def euler2rotmat(Omega: np.ndarray) -> np.ndarray:
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
def rotmat2euler(R: np.ndarray) -> np.ndarray:
    out = np.empty(3, dtype=np.double)

    val = 0.5 * ((R[0, 0] + R[1, 1] + R[2, 2]) - 1.0)
    if val > 1.0:
        val = 1.0
    elif val < -1.0:
        val = -1.0

    if val > DEF_EULER_CLOSE_TO_ONE:
        out[0] = 0.0; out[1] = 0.0; out[2] = 0.0
        return out

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
def sqrt_rot(R: np.ndarray) -> np.ndarray:
    """generates rotation matrix that corresponds to a rotation over the same axis, but over half the angle."""
    return euler2rotmat(0.5 * rotmat2euler(R))


@cond_jit(nopython=True,cache=True)
def midstep(triad1: np.ndarray, triad2: np.ndarray) -> np.ndarray:
    return triad1 @ sqrt_rot(triad1.T @ triad2)


##########################################################################################################
############### Left- and Right-Jacobians ################################################################
##########################################################################################################

@cond_jit(nopython=True,cache=True)
def right_jacobian(Omega: np.ndarray) -> np.ndarray:
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
    Omega_hat = hat_map(Omega)
    Omega_hat_sq = Omega_hat @ Omega_hat

    if Om < 1e-6:
        # Small-angle expansion: JR(Ω) ≈ I - (1/2)Ω̂ + (1/6)Ω̂²
        return np.eye(3) - 0.5 * Omega_hat + (1.0 / 6.0) * Omega_hat_sq

    Om2 = Om * Om
    Om3 = Om2 * Om

    c1 = (1.0 - np.cos(Om)) / Om2
    c2 = (Om - np.sin(Om)) / Om3

    return np.eye(3) - c1 * Omega_hat + c2 * Omega_hat_sq


@cond_jit(nopython=True,cache=True)
def left_jacobian(Omega: np.ndarray) -> np.ndarray:
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
    return right_jacobian(Omega).T


@cond_jit(nopython=True,cache=True)
def inverse_right_jacobian(Omega: np.ndarray) -> np.ndarray:
    """Compute the inverse of the right Jacobian.

    Parameters
    ----------
    Omega : ndarray, shape (3,)
        Rotation vector (axis-angle, radians).

    Returns
    -------
    JR_inv : ndarray, shape (3, 3)
        Inverse of the right Jacobian matrix.

    Notes
    -----
    For small Ω, the inverse is computed via Taylor expansion:
        JR_inv(Ω) ≈ I + (1/2)Ω̂ + (1/12)Ω̂²

    For general angles, the exact formula is:
        JR_inv(Ω) = I + (1/2)Ω̂ + b(Ω) * Ω̂²
    where b(Ω) = (c₁/2 - c₂)/(1 - c₂||Ω||²),
          c₁ = (1-cos||Ω||)/||Ω||²,
          c₂ = (||Ω|| - sin||Ω||)/||Ω||³
    """
    Om = np.linalg.norm(Omega)
    Omega_hat = hat_map(Omega)
    Omega_hat_sq = Omega_hat @ Omega_hat

    if Om < 1e-6:
        # Small-angle expansion: JR_inv(Ω) ≈ I + (1/2)Ω̂ + (1/12)Ω̂²
        return np.eye(3) + 0.5 * Omega_hat + (1.0 / 12.0) * Omega_hat_sq

    Om2 = Om * Om
    c1 = (1.0 - np.cos(Om)) / Om2
    c2 = (Om - np.sin(Om)) / (Om2 * Om)
    
    # Coefficient for Ω̂² term
    b = (c1 / 2.0 - c2) / (1.0 - c2 * Om2)

    return np.eye(3) + 0.5 * Omega_hat + b * Omega_hat_sq


@cond_jit(nopython=True,cache=True)
def inverse_left_jacobian(Omega: np.ndarray) -> np.ndarray:
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
    return inverse_right_jacobian(Omega).T


##########################################################################################################
############### SE3 Methods ##############################################################################
##########################################################################################################


@cond_jit(nopython=True,cache=True)
def se3_euler2rotmat(Omega: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    # if Omega.shape != (6,):
    #     raise ValueError(f'Expected shape (6,) array, but encountered {Omega.shape}.')
    if rotation_first:
        vrot = Omega[:3]
        vtrans = Omega[3:]
    else:
        vrot = Omega[3:]
        vtrans = Omega[:3]
    rotmat = np.zeros((4, 4))
    rotmat[:3, :3] = euler2rotmat(vrot)
    rotmat[:3, 3] = vtrans
    rotmat[3, 3] = 1
    return rotmat


@cond_jit(nopython=True,cache=True)
def se3_rotmat2euler(R: np.ndarray, rotation_first: bool = True) -> np.ndarray:
    # if R.shape != (4,4):
    #     raise ValueError(f'Expected shape (4,4) array, but encountered {R.shape}.')
    vrot = rotmat2euler(R[:3, :3])
    vtrans = R[:3, 3]
    if rotation_first:
        return np.concatenate((vrot, vtrans))
    else:
        return np.concatenate((vtrans, vrot))
    
    
def se3_eulers2rotmats(X: np.ndarray) -> np.ndarray:
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
        return se3_euler2rotmat(X)

    if X.shape[-1] != 6:
        raise ValueError(f"Expected last dimension to be 6, got {X.shape}.")

    lead_shape = X.shape[:-1]
    n = int(np.prod(lead_shape))
    X_flat = X.reshape((n, 6))

    try:
        se3_flat = se3_euler2rotmat(X_flat)
        se3_flat = np.asarray(se3_flat)
        return se3_flat.reshape(lead_shape + se3_flat.shape[-2:])
    except Exception:
        first = np.asarray(se3_euler2rotmat(X_flat[0]))
        out = np.empty((n,) + first.shape, dtype=first.dtype)
        out[0] = first
        for i in range(1, n):
            out[i] = se3_euler2rotmat(X_flat[i])
        return out.reshape(lead_shape + first.shape)

    
def se3_rotmats2eulers(se3: np.ndarray) -> np.ndarray:
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
        return se3_rotmat2euler(se3)

    lead_shape = se3.shape[:-2]
    n = int(np.prod(lead_shape))
    se3_flat = se3.reshape((n,) + se3.shape[-2:])

    try:
        X_flat = se3_rotmat2euler(se3_flat)
        X_flat = np.asarray(X_flat, dtype=np.float64)
        return X_flat.reshape(lead_shape + (6,))
    except Exception:
        X_flat = np.empty((n, 6), dtype=np.float64)
        for i in range(n):
            X_flat[i] = se3_rotmat2euler(se3_flat[i])
        return X_flat.reshape(lead_shape + (6,))

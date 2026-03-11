#!/bin/env python3

import numpy as np

from .Euler import _inverse_right_jacobian_sv, _right_jacobian_sv

from .generators import _hat_map_sv as hat_map
from ._pycondec import cond_jit


##########################################################################################################
# Internal single-vector JIT helpers
# Defined first so that the flat-batch helpers below can call them inside JIT context.
##########################################################################################################

@cond_jit(nopython=True, cache=True)
def _cayley2euler_sv(cayley: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(cayley)
    if norm < 1e-14:
        return np.zeros(3)
    return 2.0 * np.arctan(0.5 * norm) / norm * cayley


@cond_jit(nopython=True, cache=True)
def _euler2cayley_sv(euler: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(euler)
    if norm < 1e-14:
        return np.zeros(3)
    return 2.0 * np.tan(0.5 * norm) / norm * euler


@cond_jit(nopython=True, cache=True)
def _cayley2euler_factor_sv(cayley: np.ndarray) -> float:
    norm = np.linalg.norm(cayley)
    if norm < 1e-14:
        return 1.0
    return 2.0 * np.arctan(0.5 * norm) / norm


@cond_jit(nopython=True, cache=True)
def _euler2cayley_factor_sv(euler: np.ndarray) -> float:
    norm = np.linalg.norm(euler)
    if norm < 1e-14:
        return 1.0
    return 2.0 * np.tan(0.5 * norm) / norm


@cond_jit(nopython=True, cache=True)
def _cayley2euler_linearexpansion_sv(cayley_gs: np.ndarray) -> np.ndarray:
    cayley_norm = np.linalg.norm(cayley_gs)
    if cayley_norm < 1e-14:
        return np.eye(3)
    cayley_norm_sq = cayley_norm ** 2
    ratio = 2.0 * np.arctan(0.5 * cayley_norm) / cayley_norm
    fac = (4.0 / (4.0 + cayley_norm_sq) - ratio) / cayley_norm_sq
    result = np.eye(3) * ratio
    for j in range(3):
        for k in range(3):
            result[j, k] += fac * cayley_gs[j] * cayley_gs[k]
    return result


@cond_jit(nopython=True, cache=True)
def _euler2cayley_linearexpansion_sv(euler_gs: np.ndarray) -> np.ndarray:
    euler_norm = np.linalg.norm(euler_gs)
    if euler_norm < 1e-14:
        return np.eye(3)
    euler_norm_sq = euler_norm ** 2
    ratio = 2.0 * np.tan(0.5 * euler_norm) / euler_norm
    fac = (1.0 / (np.cos(0.5 * euler_norm)) ** 2 - ratio) / euler_norm_sq
    result = np.eye(3) * ratio
    for j in range(3):
        for k in range(3):
            result[j, k] += fac * euler_gs[j] * euler_gs[k]
    return result


@cond_jit(nopython=True, cache=True)
def _splittransform_group2algebra_sv(Theta_0: np.ndarray) -> np.ndarray:
    return _inverse_right_jacobian_sv(Theta_0)


@cond_jit(nopython=True, cache=True)
def _splittransform_algebra2group_sv(Theta_0: np.ndarray) -> np.ndarray:
    return _right_jacobian_sv(Theta_0)


##########################################################################################################
# Internal flat-batch JIT helpers
# Operate on pre-flattened C-contiguous (N, 3) input; return flat (N, ...) output.
##########################################################################################################

@cond_jit(nopython=True, cache=True)
def _cayley2euler_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3))
    for i in range(n):
        result[i] = _cayley2euler_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _euler2cayley_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3))
    for i in range(n):
        result[i] = _euler2cayley_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _cayley2euler_factor_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros(n)
    for i in range(n):
        result[i] = _cayley2euler_factor_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _euler2cayley_factor_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros(n)
    for i in range(n):
        result[i] = _euler2cayley_factor_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _cayley2euler_linearexpansion_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3, 3))
    for i in range(n):
        result[i] = _cayley2euler_linearexpansion_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _euler2cayley_linearexpansion_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3, 3))
    for i in range(n):
        result[i] = _euler2cayley_linearexpansion_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _splittransform_group2algebra_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3, 3))
    for i in range(n):
        result[i] = _splittransform_group2algebra_sv(flat[i])
    return result


@cond_jit(nopython=True, cache=True)
def _splittransform_algebra2group_flat(flat: np.ndarray) -> np.ndarray:
    n = flat.shape[0]
    result = np.zeros((n, 3, 3))
    for i in range(n):
        result[i] = _splittransform_algebra2group_sv(flat[i])
    return result


##########################################################################################################
# Public batch functions — shape-changing outputs
# Plain Python dispatchers; JIT inner loop is handled by the flat helpers above.
# (Cannot be JIT-compiled because return type changes between 1-D and N-D inputs.)
##########################################################################################################

def cayley2euler_batch(cayley: np.ndarray) -> np.ndarray:
    """Transforms Cayley vector(s) to corresponding Euler vector(s).

    Accepts any shape (..., 3): a single vector (3,), a batch (N, 3),
    or higher-dimensional arrays (M, N, 3), etc. The output shape matches
    the input shape.

    Args:
        cayley (np.ndarray): Cayley vector(s), last dimension must be 3.

    Returns:
        np.ndarray: Euler vector(s) with the same shape as the input.
    """
    if cayley.ndim == 1:
        return _cayley2euler_sv(cayley)
    orig_shape = cayley.shape
    n = cayley.size // 3
    flat = np.ascontiguousarray(cayley).reshape((n, 3))
    return _cayley2euler_flat(flat).reshape(orig_shape)


def euler2cayley_batch(euler: np.ndarray) -> np.ndarray:
    """Transforms Euler vector(s) to corresponding Cayley vector(s).

    Accepts any shape (..., 3): a single vector (3,), a batch (N, 3),
    or higher-dimensional arrays (M, N, 3), etc. The output shape matches
    the input shape.

    Args:
        euler (np.ndarray): Euler vector(s), last dimension must be 3.

    Returns:
        np.ndarray: Cayley vector(s) with the same shape as the input.
    """
    if euler.ndim == 1:
        return _euler2cayley_sv(euler)
    orig_shape = euler.shape
    n = euler.size // 3
    flat = np.ascontiguousarray(euler).reshape((n, 3))
    return _euler2cayley_flat(flat).reshape(orig_shape)


def cayley2euler_factor_batch(cayley: np.ndarray):
    """Conversion factor from Cayley to Euler parametrisation for one or more vectors.

    Each 3-vector is mapped to its scalar factor f such that euler = f * cayley.
    Collection dimensions are preserved: (..., 3) -> (...,).
    For a single 3-vector the return value is a Python float.

    Args:
        cayley (np.ndarray): Cayley vector(s), last dimension must be 3.

    Returns:
        float or np.ndarray: Factor(s); scalar for a single vector, shape (...,) for batch input.
    """
    cayley = np.asarray(cayley, dtype=float)
    if cayley.ndim == 1:
        return _cayley2euler_factor_sv(cayley)
    orig_shape = cayley.shape
    n = cayley.size // 3
    flat = np.ascontiguousarray(cayley).reshape((n, 3))
    return _cayley2euler_factor_flat(flat).reshape(orig_shape[:-1])


def euler2cayley_factor_batch(euler: np.ndarray):
    """Conversion factor from Euler to Cayley parametrisation for one or more vectors.

    Each 3-vector is mapped to its scalar factor f such that cayley = f * euler.
    Collection dimensions are preserved: (..., 3) -> (...,).
    For a single 3-vector the return value is a Python float.

    Args:
        euler (np.ndarray): Euler vector(s), last dimension must be 3.

    Returns:
        float or np.ndarray: Factor(s); scalar for a single vector, shape (...,) for batch input.
    """
    euler = np.asarray(euler, dtype=float)
    if euler.ndim == 1:
        return _euler2cayley_factor_sv(euler)
    orig_shape = euler.shape
    n = euler.size // 3
    flat = np.ascontiguousarray(euler).reshape((n, 3))
    return _euler2cayley_factor_flat(flat).reshape(orig_shape[:-1])


##########################################################################################################
############### Linear Transformations based on linear expansion #########################################
##########################################################################################################

def cayley2euler_linearexpansion_batch(cayley_gs: np.ndarray) -> np.ndarray:
    """Linearization of the Cayley-to-Euler transformation around a groundstate.

    Accepts a single 3-vector or arbitrary batch dimensions (..., 3).
    Output shape: (..., 3, 3).

    Args:
        cayley_gs (np.ndarray): Cayley groundstate vector(s), last dimension must be 3.

    Returns:
        np.ndarray: Jacobian matrix (matrices) of shape (..., 3, 3).
    """
    cayley_gs = np.asarray(cayley_gs, dtype=float)
    if cayley_gs.ndim == 1:
        return _cayley2euler_linearexpansion_sv(cayley_gs)
    orig_shape = cayley_gs.shape
    n = cayley_gs.size // 3
    flat = np.ascontiguousarray(cayley_gs).reshape((n, 3))
    return _cayley2euler_linearexpansion_flat(flat).reshape(orig_shape[:-1] + (3, 3))


def euler2cayley_linearexpansion_batch(euler_gs: np.ndarray) -> np.ndarray:
    """Linearization of the Euler-to-Cayley transformation around a groundstate.

    Accepts a single 3-vector or arbitrary batch dimensions (..., 3).
    Output shape: (..., 3, 3).

    Args:
        euler_gs (np.ndarray): Euler groundstate vector(s), last dimension must be 3.

    Returns:
        np.ndarray: Jacobian matrix (matrices) of shape (..., 3, 3).
    """
    euler_gs = np.asarray(euler_gs, dtype=float)
    if euler_gs.ndim == 1:
        return _euler2cayley_linearexpansion_sv(euler_gs)
    orig_shape = euler_gs.shape
    n = euler_gs.size // 3
    flat = np.ascontiguousarray(euler_gs).reshape((n, 3))
    return _euler2cayley_linearexpansion_flat(flat).reshape(orig_shape[:-1] + (3, 3))


##########################################################################################################
############### Change Splitting between static and dynamic components ###################################
##########################################################################################################

def splittransform_group2algebra_batch(Theta_0: np.ndarray) -> np.ndarray:
    """
    Linear transformation from group-splitting to algebra-splitting dynamic component.

    Maps Delta (group splitting R = exp(hat(Theta_0)) exp(hat(Delta))) to
    Delta' (algebra splitting R = exp(hat(Theta_0) + hat(Delta'))) via T*Delta = Delta'.

    Accepts a single 3-vector or arbitrary batch dimensions (..., 3).
    Output shape: (..., 3, 3).

    Args:
        Theta_0 (np.ndarray): Static Euler vector(s) in radians, last dimension must be 3.

    Returns:
        np.ndarray: Transformation matrix T of shape (..., 3, 3).
    """
    Theta_0 = np.asarray(Theta_0, dtype=float)
    if Theta_0.ndim == 1:
        return _splittransform_group2algebra_sv(Theta_0)
    orig_shape = Theta_0.shape
    n = Theta_0.size // 3
    flat = np.ascontiguousarray(Theta_0).reshape((n, 3))
    return _splittransform_group2algebra_flat(flat).reshape(orig_shape[:-1] + (3, 3))


def splittransform_algebra2group_batch(Theta_0: np.ndarray) -> np.ndarray:
    """
    Linear transformation from algebra-splitting to group-splitting dynamic component.

    Inverse of splittransform_group2algebra. Maps Delta' to Delta via T'*Delta' = Delta.

    Accepts a single 3-vector or arbitrary batch dimensions (..., 3).
    Output shape: (..., 3, 3).

    Args:
        Theta_0 (np.ndarray): Static Euler vector(s) in radians, last dimension must be 3.

    Returns:
        np.ndarray: Transformation matrix T' of shape (..., 3, 3).
    """
    Theta_0 = np.asarray(Theta_0, dtype=float)
    if Theta_0.ndim == 1:
        return _splittransform_algebra2group_sv(Theta_0)
    orig_shape = Theta_0.shape
    n = Theta_0.size // 3
    flat = np.ascontiguousarray(Theta_0).reshape((n, 3))
    return _splittransform_algebra2group_flat(flat).reshape(orig_shape[:-1] + (3, 3))


##########################################################################################################
# Public single-vector API
# Aliases to the internal JIT helpers; retained for calling from numba-compiled code.
# All _single functions accept a single (3,) vector and are JIT-callable.
##########################################################################################################

# #: Single-vector version of :func:`cayley2euler`. Returns (3,). JIT-callable.
# cayley2euler_single = _cayley2euler_sv

# #: Single-vector version of :func:`euler2cayley`. Returns (3,). JIT-callable.
# euler2cayley_single = _euler2cayley_sv

# #: Single-vector version of :func:`cayley2euler_factor`. Returns float. JIT-callable.
# cayley2euler_factor_single = _cayley2euler_factor_sv

# #: Single-vector version of :func:`euler2cayley_factor`. Returns float. JIT-callable.
# euler2cayley_factor_single = _euler2cayley_factor_sv

# #: Single-vector version of :func:`cayley2euler_linearexpansion`. Returns (3, 3). JIT-callable.
# cayley2euler_linearexpansion_single = _cayley2euler_linearexpansion_sv

# #: Single-vector version of :func:`euler2cayley_linearexpansion`. Returns (3, 3). JIT-callable.
# euler2cayley_linearexpansion_single = _euler2cayley_linearexpansion_sv

# #: Single-vector version of :func:`splittransform_group2algebra`. Returns (3, 3). JIT-callable.
# splittransform_group2algebra_single = _splittransform_group2algebra_sv

# #: Single-vector version of :func:`splittransform_algebra2group`. Returns (3, 3). JIT-callable.
# splittransform_algebra2group_single = _splittransform_algebra2group_sv



@cond_jit(nopython=True,cache=True)
def cayley2euler_single(cayley: np.ndarray) -> np.ndarray:
    """Transforms Cayley vector to corresponding Euler vector

    Args:
        cayley (np.ndarray): Cayley vector (3-vector)

    Returns:
        np.ndarray: Euler vector (3-vector)
    """
    norm = np.linalg.norm(cayley)
    if np.abs(norm) < 1e-14:
        return np.zeros(3)
    return 2 * np.arctan(0.5 * norm) / norm * cayley


@cond_jit(nopython=True,cache=True)
def cayley2euler_factor_single(cayley: np.ndarray) -> float:
    """Transforms Cayley vector to corresponding Euler vector

    Args:
        cayley (np.ndarray): Cayley vector (3-vector)

    Returns:
        float: conversion factor
    """
    norm = np.linalg.norm(cayley)
    if np.abs(norm) < 1e-14:
        return 1.0
    return 2 * np.arctan(0.5 * norm) / norm


@cond_jit(nopython=True,cache=True)
def euler2cayley_single(euler: np.ndarray) -> np.ndarray:
    """Transforms Euler vector to corresponding Cayley vector

    Args:
        euler (np.ndarray): Euler vector (3-vector)

    Returns:
        np.ndarray: Cayley vector (3-vector)
    """
    norm = np.linalg.norm(euler)
    if np.abs(norm) < 1e-14:
        return np.zeros(3)
    return 2 * np.tan(0.5 * norm) / norm * euler


@cond_jit(nopython=True,cache=True)
def euler2cayley_factor_single(euler: np.ndarray) -> float:
    """Transforms Euler vector to corresponding Cayley vector

    Args:
        euler (np.ndarray): Euler vector (3-vector)

    Returns:
        float: conversion factor
    """
    norm = np.linalg.norm(euler)
    if np.abs(norm) < 1e-14:
        return 1.0
    return 2 * np.tan(0.5 * norm) / norm


##########################################################################################################
############### Linear Transformations based on linear expansion #########################################
##########################################################################################################


@cond_jit(nopython=True,cache=True)
def cayley2euler_linearexpansion_single(cayley_gs: np.ndarray) -> np.ndarray:
    """Linearization of the transformation from Cayley to Euler vector around a given groundstate vector

    Args:
        cayley_gs (np.ndarray): The Cayley vector around which the transformation is linearly expanded

    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    cayley_norm = np.linalg.norm(cayley_gs)
    if cayley_norm < 1e-14:
        return np.eye(3)
    cayley_norm_sq = cayley_norm**2
    ratio_euler_cayley = 2 * np.arctan(0.5 * cayley_norm) / cayley_norm
    fac = (4.0 / (4 + cayley_norm_sq) - ratio_euler_cayley) / cayley_norm_sq
    return np.eye(3) * ratio_euler_cayley + np.outer(cayley_gs, cayley_gs) * fac


@cond_jit(nopython=True,cache=True)
def euler2cayley_linearexpansion_single(euler_gs: np.ndarray) -> np.ndarray:
    """Linearization of the transformation from Euler to Cayley vector around a given groundstate vector

    Args:
        cayley_gs (np.ndarray): The Euler vector around which the transformation is linearly expanded

    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    euler_norm = np.linalg.norm(euler_gs)
    if euler_norm < 1e-14:
        return np.eye(3)
    euler_norm_sq = euler_norm**2
    ratio_cayley_euler = 2 * np.tan(0.5 * euler_norm) / euler_norm
    fac = (1.0 / (np.cos(0.5 * euler_norm)) ** 2 - ratio_cayley_euler) / euler_norm_sq
    return np.eye(3) * ratio_cayley_euler + np.outer(euler_gs, euler_gs) * fac


##########################################################################################################
############### Change Splitting between static and dynamic components ###################################
##########################################################################################################


@cond_jit(nopython=True,cache=True)
def splittransform_group2algebra_single(Theta_0: np.ndarray) -> np.ndarray:
    """
    Linear transformation that maps dynamic component in group splitting representation
    (R = D*S = exp(hat(Theta_0))exp(hat(Delta))), with D,S in SO(3) to lie algebra splitting
    representation R = exp(hat(Theta_0) + hat(Delta')). Linear transformation T transforms Delta
    into Delta' as T*Delta = Delta'.

    Args:
        Theta_0 (np.ndarray): static rotational component expressed in Axis angle parametrization (Euler vector)
        Has to be expressed in radians

    Returns:
        float: Linear transformation matrix T (3x3) that transforms Delta into Delta': T*Delta = Delta'
    """
    return _inverse_right_jacobian_sv(Theta_0)
    
    # htheta = hat_map(Theta_0)
    # hthetasq = np.dot(htheta, htheta)

    # accutheta = np.copy(htheta)
    # # zeroth order
    # T = np.eye(3)
    # # first order
    # T += 0.5 * accutheta
    # # second order
    # accutheta = np.dot(accutheta, htheta)
    # T += 1.0 / 12 * accutheta
    # # fourth order
    # accutheta = np.dot(accutheta, hthetasq)
    # T += -1.0 / 720 * accutheta
    # # sixth order
    # accutheta = np.dot(accutheta, hthetasq)
    # T += 1.0 / 30240 * accutheta
    # # eighth order
    # accutheta = np.dot(accutheta, hthetasq)
    # T += -1.0 / 1209600 * accutheta
    # # tenth order
    # accutheta = np.dot(accutheta, hthetasq)
    # T += 1.0 / 47900160 * accutheta
    # # twelth order
    # accutheta = np.dot(accutheta, hthetasq)
    # T += -691.0 / 1307674368000 * accutheta
    # return T


@cond_jit(nopython=True,cache=True)
def splittransform_algebra2group_single(Theta_0: np.ndarray) -> np.ndarray:
    """
    Linear transformation that maps dynamic component in lie algebra splitting representation R = exp(hat(Theta_0) + hat(Delta')) to group splitting representation
    (R = D*S = exp(hat(Theta_0))exp(hat(Delta))), with D,S in SO(3) t. Linear transformation T transforms Delta
    into Delta' as T'*Delta' = Delta. Currently this is defined as the inverse of the transformation
    defined in the method splittransform_group2algebra

    Args:
        Theta_0 (np.ndarray): static rotational component expressed in Axis angle parametrization (Euler vector)
        Has to be expressed in radians

    Returns:
        float: Linear transformation matrix T' (3x3) that transforms Delta into Delta': T'*Delta' = Delta
    """
    
    return _right_jacobian_sv(Theta_0)
    # return np.linalg.inv(splittransform_group2algebra(Theta_0))

#!/bin/env python3

import numpy as np

from .Euler import inverse_right_jacobian, right_jacobian

from .generators import hat_map
from ._pycondec import cond_jit

# from .generators import hat_map, vec_map, generator1, generator2, generator3


@cond_jit(nopython=True,cache=True)
def _cayley2euler_single(cayley: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(cayley)
    if np.abs(norm) < 1e-14:
        return np.zeros(3)
    return 2 * np.arctan(0.5 * norm) / norm * cayley


@cond_jit(nopython=True,cache=True)
def _cayley2euler_batch(cayley: np.ndarray) -> np.ndarray:
    result = np.zeros_like(cayley)
    for i in range(cayley.shape[0]):
        norm = np.linalg.norm(cayley[i])
        if np.abs(norm) >= 1e-14:
            fac = 2.0 * np.arctan(0.5 * norm) / norm
            result[i, 0] = fac * cayley[i, 0]
            result[i, 1] = fac * cayley[i, 1]
            result[i, 2] = fac * cayley[i, 2]
    return result


@cond_jit(nopython=True,cache=True)
def cayley2euler(cayley: np.ndarray) -> np.ndarray:
    """Transforms Cayley vector(s) to corresponding Euler vector(s).

    Args:
        cayley (np.ndarray): Cayley vector (3-vector) or array of Cayley vectors (Nx3)

    Returns:
        np.ndarray: Euler vector (3-vector) or array of Euler vectors (Nx3)
    """
    if cayley.ndim == 1:
        return _cayley2euler_single(cayley)
    return _cayley2euler_batch(cayley)


@cond_jit(nopython=True,cache=True)
def cayley2euler_factor(cayley: np.ndarray) -> float:
    """Transforms Cayley vector to corresponding Euler vector

    Args:
        cayley (np.ndarray): Cayley vector (3-vector)

    Returns:
        float: conversion factor
    """
    norm = np.linalg.norm(cayley)
    if np.abs(norm) < 1e-14:
        return 0.0
    return 2 * np.arctan(0.5 * norm) / norm


@cond_jit(nopython=True,cache=True)
def _euler2cayley_single(euler: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(euler)
    if np.abs(norm) < 1e-14:
        return np.zeros(3)
    return 2 * np.tan(0.5 * norm) / norm * euler


@cond_jit(nopython=True,cache=True)
def _euler2cayley_batch(euler: np.ndarray) -> np.ndarray:
    result = np.zeros_like(euler)
    for i in range(euler.shape[0]):
        norm = np.linalg.norm(euler[i])
        if np.abs(norm) >= 1e-14:
            fac = 2.0 * np.tan(0.5 * norm) / norm
            result[i, 0] = fac * euler[i, 0]
            result[i, 1] = fac * euler[i, 1]
            result[i, 2] = fac * euler[i, 2]
    return result


@cond_jit(nopython=True,cache=True)
def euler2cayley(euler: np.ndarray) -> np.ndarray:
    """Transforms Euler vector(s) to corresponding Cayley vector(s).

    Args:
        euler (np.ndarray): Euler vector (3-vector) or array of Euler vectors (Nx3)

    Returns:
        np.ndarray: Cayley vector (3-vector) or array of Cayley vectors (Nx3)
    """
    if euler.ndim == 1:
        return _euler2cayley_single(euler)
    return _euler2cayley_batch(euler)


@cond_jit(nopython=True,cache=True)
def euler2cayley_factor(euler: np.ndarray) -> float:
    """Transforms Euler vector to corresponding Cayley vector

    Args:
        euler (np.ndarray): Euler vector (3-vector)

    Returns:
        float: conversion factor
    """
    norm = np.linalg.norm(euler)
    if np.abs(norm) < 1e-14:
        return 0.0
    return 2 * np.tan(0.5 * norm) / norm


##########################################################################################################
############### Linear Transformations based on linear expansion #########################################
##########################################################################################################


@cond_jit(nopython=True,cache=True)
def cayley2euler_linearexpansion(cayley_gs: np.ndarray) -> np.ndarray:
    """Linearization of the transformation from Cayley to Euler vector around a given groundstate vector

    Args:
        cayley_gs (np.ndarray): The Cayley vector around which the transformation is linearly expanded

    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    cayley_norm = np.linalg.norm(cayley_gs)
    cayley_norm_sq = cayley_norm**2
    ratio_euler_cayley = 2 * np.arctan(0.5 * cayley_norm) / cayley_norm
    fac = (4.0 / (4 + cayley_norm_sq) - ratio_euler_cayley) / cayley_norm_sq
    return np.eye(3) * ratio_euler_cayley + np.outer(cayley_gs, cayley_gs) * fac


@cond_jit(nopython=True,cache=True)
def euler2cayley_linearexpansion(euler_gs: np.ndarray) -> np.ndarray:
    """Linearization of the transformation from Euler to Cayley vector around a given groundstate vector

    Args:
        cayley_gs (np.ndarray): The Euler vector around which the transformation is linearly expanded

    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    euler_norm = np.linalg.norm(euler_gs)
    euler_norm_sq = euler_norm**2
    ratio_cayley_euler = 2 * np.tan(0.5 * euler_norm) / euler_norm
    fac = (1.0 / (np.cos(0.5 * euler_norm)) ** 2 - ratio_cayley_euler) / euler_norm_sq
    return np.eye(3) * ratio_cayley_euler + np.outer(euler_gs, euler_gs) * fac


##########################################################################################################
############### Change Splitting between static and dynamic components ###################################
##########################################################################################################


@cond_jit(nopython=True,cache=True)
def splittransform_group2algebra(Theta_0: np.ndarray) -> np.ndarray:
    """
    Linear transformation that maps dynamic component in group splitting representation
    (R = D*S = exp(hat(Theta_0))exp(hat(Delta))), with D,S \in SO(3) to lie algebra splitting
    representation R = exp(hat(Theta_0) + hat(Delta')). Linear transformation T transforms Delta
    into Delta' as T*Delta = Delta'.

    Args:
        Theta_0 (np.ndarray): static rotational component expressed in Axis angle parametrization (Euler vector)
        Has to be expressed in radians

    Returns:
        float: Linear transformation matrix T (3x3) that transforms Delta into Delta': T*Delta = Delta'
    """
    return inverse_right_jacobian(Theta_0)
    
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
def splittransform_algebra2group(Theta_0: np.ndarray) -> np.ndarray:
    """
    Linear transformation that maps dynamic component in lie algebra splitting representation R = exp(hat(Theta_0) + hat(Delta')) to group splitting representation
    (R = D*S = exp(hat(Theta_0))exp(hat(Delta))), with D,S \in SO(3) t. Linear transformation T transforms Delta
    into Delta' as T'*Delta' = Delta. Currently this is defined as the inverse of the transformation
    defined in the method splittransform_group2algebra

    Args:
        Theta_0 (np.ndarray): static rotational component expressed in Axis angle parametrization (Euler vector)
        Has to be expressed in radians

    Returns:
        float: Linear transformation matrix T' (3x3) that transforms Delta into Delta': T'*Delta' = Delta
    """
    
    return right_jacobian(Theta_0)
    # return np.linalg.inv(splittransform_group2algebra(Theta_0))

#!/bin/env python3

import numpy as np
from .pyConDec.pycondec import cond_jit
# from .generators import hat_map, vec_map, generator1, generator2, generator3


@cond_jit
def cayley2euler(cayley: np.ndarray) -> np.ndarray: 
    """Transforms Cayley vector to corresponding Euler vector

    Args:
        cayley (np.ndarray): Cayley vector (3-vector)

    Returns:
        np.ndarray: Euler vector (3-vector)
    """
    norm = np.linalg.norm(cayley)
    return 2*np.arctan(0.5*norm)/norm * cayley


@cond_jit
def cayley2euler_factor(cayley: np.ndarray) -> float: 
    """Transforms Cayley vector to corresponding Euler vector

    Args:
        cayley (np.ndarray): Cayley vector (3-vector)

    Returns:
        float: conversion factor
    """
    norm = np.linalg.norm(cayley)
    return 2*np.arctan(0.5*norm)/norm

@cond_jit
def euler2cayley(euler: np.ndarray) -> np.ndarray: 
    """Transforms Euler vector to corresponding Cayley vector

    Args:
        euler (np.ndarray): Euler vector (3-vector)

    Returns:
        np.ndarray: Cayley vector (3-vector)
    """
    norm = np.linalg.norm(euler)
    return 2*np.tan(0.5*norm)/norm * euler

@cond_jit
def euler2cayley_factor(euler: np.ndarray) -> float: 
    """Transforms Euler vector to corresponding Cayley vector

    Args:
        euler (np.ndarray): Euler vector (3-vector)

    Returns:
        float: conversion factor
    """
    norm = np.linalg.norm(euler)
    return 2*np.tan(0.5*norm)/norm

##########################################################################################################
############### Linear Transformations based on linear expansion #########################################
##########################################################################################################

def cayley2euler_linearexpansion(cayley_gs: np.ndarray) -> np.ndarray:
    """Linearization of the transformation from Cayley to Euler vector around a given groundstate vector

    Args:
        cayley_gs (np.ndarray): The Cayley vector around which the transformation is linearly expanded

    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    cnorm = np.linalg.norm(cayley_gs)
    csq = cnorm**2
    fac = 2./csq*(2./(4+csq) - np.arctan(cnorm/2)/cnorm)
    mat = np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            mat[i,j] = fac * cayley_gs[i] * cayley_gs[j]
        mat[i,i] += 2*np.arctan(0.5*cnorm)/cnorm
    return mat


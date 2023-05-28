#!/bin/env python3

import numpy as np
from .pyConDec.pycondec import cond_jit

from .generators import hat_map, vec_map, generator1, generator2, generator3

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


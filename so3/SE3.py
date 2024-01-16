#!/bin/env python3

import numpy as np
from typing import List
from .pyConDec.pycondec import cond_jit
from .Euler import euler2rotmat, rotmat2euler

@cond_jit
def se3_inverse(g: np.ndarray) -> np.ndarray:
    inv = np.zeros(g.shape)
    inv[:3,:3] = g[:3,:3].T
    inv[:3,3]  = -inv[:3,:3]@g[:3,3]
    inv[3,3] = 1
    return inv

@cond_jit
def se3_triads2rotmat(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    return se3_inverse(tau1)@tau2

@cond_jit
def se3_triadxrotmat_midsteptrans(tau1: np.ndarray, g: np.ndarray) -> np.ndarray:
    R = g[:3,:3]
    T1 = tau1[:3,:3]
    tau2 = np.eye(4)
    tau2[:3,:3] = T1 @ R
    tau2[:3,3] = tau1[:3,3] + T1 @ sqrt_rot(R) @ g[:3,3]
    return tau2

@cond_jit
def se3_triads2rotmat_midsteptrans(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    T1 = tau1[:3,:3]
    T2 = tau2[:3,:3]
    R = T1.T @ T2
    Tmid = T1 @ sqrt_rot(R)
    zeta = Tmid.T @ (tau2[:3,3]-tau1[:3,3])
    g = np.ones(4)
    g[:3,:3] = R
    g[:3,3]  = zeta
    return g

@cond_jit
def sqrt_rot(R: np.ndarray) -> np.ndarray:
    return euler2rotmat(0.5*rotmat2euler(R))
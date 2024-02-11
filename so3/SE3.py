#!/bin/env python3

import numpy as np
from typing import List
from .pyConDec.pycondec import cond_jit
from .Euler import euler2rotmat, rotmat2euler, se3_rotmat2euler

@cond_jit
def se3_inverse(g: np.ndarray) -> np.ndarray:
    """Inverse of element of SE3
    """
    inv = np.zeros(g.shape)
    inv[:3,:3] = g[:3,:3].T
    inv[:3,3]  = -inv[:3,:3]@g[:3,3]
    inv[3,3] = 1
    return inv

@cond_jit
def se3_triads2rotmat(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """find SE3 transformation matrix, g, that maps tau1 into tau2 with respect to the frame of tau1
    """
    return se3_inverse(tau1)@tau2

@cond_jit
def se3_triads2euler(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    return se3_rotmat2euler(se3_triads2rotmat(tau1,tau2))

@cond_jit
def se3_triad2midstep(midstep_euler: np.ndarray) -> np.ndarray:
    triad_euler = np.copy(midstep_euler)
    vrot = midstep_euler[:3]
    vtrans = midstep_euler[3:]
    sqrt_rotmat = euler2rotmat(0.5*vrot)
    triad_euler[3:] = sqrt_rotmat @ vtrans
    return triad_euler

@cond_jit
def se3_midstep2triad(triad_euler: np.ndarray) -> np.ndarray:
    midstep_euler = np.copy(triad_euler)
    vrot = triad_euler[:3]
    vtrans = triad_euler[3:]
    sqrt_rotmat = euler2rotmat(0.5*vrot)
    midstep_euler[3:] = sqrt_rotmat.T @ vtrans
    return midstep_euler

@cond_jit
def se3_triadxrotmat_midsteptrans(tau1: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Multiplication of triad with rotation matrix g (in SE3) assuming that the translation of g is defined with respect to the midstep triad.
    """
    R = g[:3,:3]
    T1 = tau1[:3,:3]
    tau2 = np.eye(4)
    tau2[:3,:3] = T1 @ R
    tau2[:3,3] = tau1[:3,3] + T1 @ sqrt_rot(R) @ g[:3,3]
    return tau2

@cond_jit
def se3_triads2rotmat_midsteptrans(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """find SE3 transformation matrix, g, that maps tau1 into tau2 with respect to the frame of tau1, assuming that the translation of g is defined with respect to the midstep triad.
    """
    T1 = tau1[:3,:3]
    T2 = tau2[:3,:3]
    R = T1.T @ T2
    Tmid = T1 @ sqrt_rot(R)
    zeta = Tmid.T @ (tau2[:3,3]-tau1[:3,3])
    g = np.eye(4)
    g[:3,:3] = R
    g[:3,3]  = zeta
    return g

@cond_jit
def se3_transformation_triad2midstep(g: np.ndarray) -> np.ndarray:
    """transforms translation of transformation g (in SE3) from canonical definition to mid-step triad definition.
    """
    midg = np.copy(g)
    midg[:3,3] = np.transpose(sqrt_rot(g[:3,:3])) @ g[:3,3]
    return midg

@cond_jit
def se3_transformation_midstep2triad(midg: np.ndarray) -> np.ndarray:
    """transforms translation of transformation g (in SE3) from mid-step triad definition to canonical definition.
    """
    g = np.copy(midg)
    g[:3,3] = sqrt_rot(midg[:3,:3]) @ midg[:3,3]
    return g

@cond_jit
def sqrt_rot(R: np.ndarray) -> np.ndarray:
    """generates rotation matrix that corresponds to a rotation over the same axis, but over half the angle.
    """
    return euler2rotmat(0.5*rotmat2euler(R))
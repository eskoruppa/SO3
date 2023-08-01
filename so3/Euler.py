#!/bin/env python3

import numpy as np
from .pyConDec.pycondec import cond_jit


DEF_EULER_EPSILON             =  1e-12
DEF_EULER_CLOSE_TO_ONE        =  0.999999999999
DEF_EULER_CLOSE_TO_MINUS_ONE  = -0.999999999999

# @cond_jit
def euler2rotmat(Omega: np.ndarray) -> np.ndarray:
    """Returns the matrix version of the Euler-Rodrigues formula

    Args:
        Omega (np.ndarray): Euler vector / Rotation vector (3-vector)

    Returns:
        np.ndarray: Rotation matrix (element of SO(3))
    """
    Om = np.linalg.norm(Omega)
    R = np.zeros((3,3),dtype=np.double)
    
    # if norm is zero, return identity matrix
    if Om<DEF_EULER_EPSILON:
        R[0,0] = 1
        R[1,1] = 1
        R[2,2] = 1
        return R
    
    cosOm = np.cos(Om)
    sinOm = np.sin(Om)
    Omsq  = Om*Om

    R[0,0] = cosOm+Omega[0]**2/Omsq*(1-cosOm)
    R[1,1] = cosOm+Omega[1]**2/Omsq*(1-cosOm)
    R[2,2] = cosOm+Omega[2]**2/Omsq*(1-cosOm)
    A = Omega[0]*Omega[1]/Omsq*(1-cosOm)
    B = Omega[2]/Om*sinOm
    R[0,1] = A-B
    R[1,0] = A+B
    A = Omega[0]*Omega[2]/Omsq*(1-cosOm)
    B = Omega[1]/Om*sinOm
    R[0,2] = A+B
    R[2,0] = A-B
    A = Omega[1]*Omega[2]/Omsq*(1-cosOm)
    B = Omega[0]/Om*sinOm
    R[1,2] = A-B
    R[2,1] = A+B
    return R

# @cond_jit
def rotmat2euler(R: np.ndarray) -> np.ndarray:
    """Inversion of Euler Rodriguez Formula

    Args:
        R (np.ndarray): Rotation matrix (element of SO(3))

    Returns:
        np.ndarray: Euler vector / Rotation vector (3-vector)
    """
    val = 0.5*(np.trace(R)-1)
    if (val > DEF_EULER_CLOSE_TO_ONE):
        return np.zeros(3)
    if (val < DEF_EULER_CLOSE_TO_MINUS_ONE):
        if (R[0,0] > DEF_EULER_CLOSE_TO_ONE):
            return np.array([np.pi,0,0])
        if (R[1,1] > DEF_EULER_CLOSE_TO_ONE):
            return np.array([0,np.pi,0])
        return np.array([0,0,np.pi])
    Th = np.arccos(val)
    Theta =  np.array([(R[2,1]-R[1,2]),(R[0,2]-R[2,0]),(R[1,0]-R[0,1])])
    Theta = Th*0.5/np.sin(Th) * Theta
    return Theta
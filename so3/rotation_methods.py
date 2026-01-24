#!/bin/env python3

import numpy as np
from .Euler import euler2rotmat
from .pyConDec.pycondec import cond_jit


@cond_jit(nopython=True,cache=True)
def rotmat_align_vector(vec_from: np.ndarray, vec_to: np.ndarray) -> np.ndarray:
    """Compute rotation matrix to align vec_from to vec_to.
    
    Constructs the rotation matrix R such that R @ vec_from is parallel to vec_to,
    using the axis-angle representation via Rodrigues' rotation formula.
    
    Parameters
    ----------
    vec_from : np.ndarray
        Source vector (3D). Will be normalized internally.
    vec_to : np.ndarray
        Target vector (3D). Will be normalized internally.
    
    Returns
    -------
    np.ndarray
        3x3 rotation matrix that rotates vec_from to align with vec_to.
    
    Notes
    -----
    - For aligned vectors (dot product ≈ 1), returns identity matrix.
    - For anti-parallel vectors (dot product ≈ -1), rotates 180° around an 
      arbitrary perpendicular axis.
    - For general case, rotates around the axis n = vec_from × vec_to by angle 
      θ = arccos(vec_from · vec_to).
    """
    vec_from = vec_from / np.linalg.norm(vec_from)
    vec_to = vec_to / np.linalg.norm(vec_to)
    dot = vec_from.dot(vec_to)
    
    # Already aligned
    if dot > 0.99999999:
        return np.eye(3)
    
    # Anti-parallel case: rotate 180° around any perpendicular axis
    if dot < -0.99999999:
        # Find a perpendicular vector
        if abs(vec_from[0]) < 0.9:
            perp = np.cross(vec_from, np.array([1.0, 0.0, 0.0]))
        else:
            perp = np.cross(vec_from, np.array([0.0, 1.0, 0.0]))
        perp = perp / np.linalg.norm(perp)
        Theta = np.pi * perp
        return euler2rotmat(Theta)
    
    # General case
    n = np.cross(vec_from, vec_to)
    n = n / np.linalg.norm(n)
    # Clamp dot product to avoid numerical errors in arccos
    dot_clamped = max(-1.0, min(1.0, dot))
    angle = np.arccos(dot_clamped)
    Theta = angle * n
    return euler2rotmat(Theta)  
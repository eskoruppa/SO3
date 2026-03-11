#!/bin/env python3

from typing import List

import numpy as np

from .conversions import _splittransform_algebra2group_sv, _splittransform_group2algebra_sv
from .Euler import _euler2rotmat_sv, _sqrt_rot_sv, _se3_rotmat2euler_sv
from .generators import _hat_map_sv
from ._pycondec import cond_jit

from .Euler import euler2rotmat_single, sqrt_rot_single, se3_rotmat2euler_single
from .conversions import splittransform_algebra2group_single, splittransform_group2algebra_single
from .generators import hat_map_single


##############################################################################
# Layer 1 – single-input JIT helpers (_sv)
##############################################################################

@cond_jit(nopython=True,cache=True)
def _se3_inverse_sv(g: np.ndarray) -> np.ndarray:
    """Inverse of element of SE3"""
    inv = np.empty(g.shape, dtype=g.dtype)
    inv[:3, :3] = g[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ g[:3, 3]
    inv[3, :]  = np.array([0, 0, 0, 1])
    return inv


@cond_jit(nopython=True,cache=True)
def _se3_triads2rotmat_sv(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """find SE3 transformation matrix, g, that maps tau1 into tau2 with respect to the frame of tau1"""
    return _se3_inverse_sv(tau1) @ tau2


@cond_jit(nopython=True,cache=True)
def _se3_triads2euler_sv(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    return _se3_rotmat2euler_sv(_se3_triads2rotmat_sv(tau1, tau2))

@cond_jit(nopython=True,cache=True)
def _se3_midstep2triad_sv(midstep_euler: np.ndarray) -> np.ndarray:
    triad_euler = np.copy(midstep_euler)
    vrot = midstep_euler[:3]
    vtrans = midstep_euler[3:]
    sqrt_rotmat = _euler2rotmat_sv(0.5 * vrot)
    triad_euler[3:] = sqrt_rotmat @ vtrans
    return triad_euler

@cond_jit(nopython=True,cache=True)
def _se3_triad2midstep_sv(triad_euler: np.ndarray) -> np.ndarray:
    midstep_euler = np.copy(triad_euler)
    vrot = triad_euler[:3]
    vtrans = triad_euler[3:]
    sqrt_rotmat = _euler2rotmat_sv(0.5 * vrot)
    midstep_euler[3:] = sqrt_rotmat.T @ vtrans
    return midstep_euler

@cond_jit(nopython=True,cache=True)
def _se3_triadxrotmat_midsteptrans_sv(tau1: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Multiplication of triad with rotation matrix g (in SE3) assuming that the translation of g is defined with respect to the midstep triad."""
    R = g[:3, :3]
    T1 = tau1[:3, :3]
    tau2 = np.eye(4)
    tau2[:3, :3] = T1 @ R
    tau2[:3, 3] = tau1[:3, 3] + T1 @ _sqrt_rot_sv(R) @ g[:3, 3]
    return tau2


@cond_jit(nopython=True,cache=True)
def _se3_triads2rotmat_midsteptrans_sv(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """find SE3 transformation matrix, g, that maps tau1 into tau2 with respect to the frame of tau1, assuming that the translation of g is defined with respect to the midstep triad."""
    T1 = tau1[:3, :3]
    T2 = tau2[:3, :3]
    R = T1.T @ T2
    Tmid = T1 @ _sqrt_rot_sv(R)
    zeta = Tmid.T @ (tau2[:3, 3] - tau1[:3, 3])
    g = np.eye(4)
    g[:3, :3] = R
    g[:3, 3] = zeta
    return g


@cond_jit(nopython=True,cache=True)
def _se3_transformation_triad2midstep_sv(g: np.ndarray) -> np.ndarray:
    """transforms translation of transformation g (in SE3) from canonical definition to mid-step triad definition."""
    midg = np.copy(g)
    midg[:3, 3] = np.transpose(_sqrt_rot_sv(g[:3, :3])) @ g[:3, 3]
    return midg


@cond_jit(nopython=True,cache=True)
def _se3_transformation_midstep2triad_sv(midg: np.ndarray) -> np.ndarray:
    """transforms translation of transformation g (in SE3) from mid-step triad definition to canonical definition."""
    g = np.copy(midg)
    g[:3, 3] = _sqrt_rot_sv(midg[:3, :3]) @ midg[:3, 3]
    return g


@cond_jit(nopython=True,cache=True)
def _se3_algebra2group_lintrans_sv(
    groundstate_algebra: np.ndarray, translation_as_midstep: bool = False
) -> np.ndarray:
    Trans = np.eye(6)
    Omega_0 = groundstate_algebra[:3]
    zeta_0 = groundstate_algebra[3:]

    Trans[:3, :3] = _splittransform_algebra2group_sv(Omega_0)
    if translation_as_midstep:
        sqrtS_transp = _euler2rotmat_sv(-0.5 * Omega_0)
        zeta_0_hat_transp = _hat_map_sv(-zeta_0)
        H_half = _splittransform_algebra2group_sv(0.5 * Omega_0)
        Trans[3:, :3] = 0.5 * sqrtS_transp @ zeta_0_hat_transp @ H_half
        Trans[3:, 3:] = sqrtS_transp
    else:
        Trans[3:, 3:] = _euler2rotmat_sv(-Omega_0)
    return Trans


@cond_jit(nopython=True,cache=True)
def _se3_group2algebra_lintrans_sv(
    groundstate_group: np.ndarray, translation_as_midstep: bool = False
) -> np.ndarray:
    Trans = np.eye(6)
    Phi_0 = groundstate_group[:3]
    s = groundstate_group[3:]

    H_inv = _splittransform_group2algebra_sv(Phi_0)
    Trans[:3, :3] = H_inv
    if translation_as_midstep:
        sqrtS = _euler2rotmat_sv(0.5 * Phi_0)
        zeta_0 = sqrtS.T @ s
        zeta_0_hat_transp = _hat_map_sv(-zeta_0)
        H_half = _splittransform_algebra2group_sv(0.5 * Phi_0)
        Trans[3:, :3] = -0.5 * zeta_0_hat_transp @ H_half @ H_inv
        Trans[3:, 3:] = sqrtS
    else:
        Trans[3:, 3:] = _euler2rotmat_sv(Phi_0)
    return Trans


@cond_jit(nopython=True,cache=True)
def _se3_algebra2group_stiffmat_sv(
    groundstate_algebra: np.ndarray,
    stiff_algebra: np.ndarray,
    translation_as_midstep: bool = False,
) -> np.ndarray:
    HX = _se3_algebra2group_lintrans_sv(
        groundstate_algebra, translation_as_midstep
    )
    HX_inv = np.linalg.inv(HX)
    return HX_inv.T @ stiff_algebra @ HX_inv


@cond_jit(nopython=True,cache=True)
def _se3_group2algebra_stiffmat_sv(
    groundstate_group: np.ndarray,
    stiff_group: np.ndarray,
    translation_as_midstep: bool = False,
) -> np.ndarray:
    HX_inv = _se3_group2algebra_lintrans_sv(
        groundstate_group, translation_as_midstep
    )
    HX = np.linalg.inv(HX_inv)
    return HX.T @ stiff_group @ HX


##############################################################################
# Layer 2 – flat-batch JIT helpers (_flat)
##############################################################################

@cond_jit(nopython=True,cache=True)
def _se3_inverse_flat(flat: np.ndarray) -> np.ndarray:
    N = flat.shape[0]
    result = np.empty((N, 4, 4), dtype=flat.dtype)
    for i in range(N):
        result[i] = _se3_inverse_sv(flat[i])
    return result


@cond_jit(nopython=True,cache=True)
def _se3_triads2rotmat_flat(flat1: np.ndarray, flat2: np.ndarray) -> np.ndarray:
    N = flat1.shape[0]
    result = np.empty((N, 4, 4), dtype=flat1.dtype)
    for i in range(N):
        result[i] = _se3_triads2rotmat_sv(flat1[i], flat2[i])
    return result


@cond_jit(nopython=True,cache=True)
def _se3_triads2euler_flat(flat1: np.ndarray, flat2: np.ndarray) -> np.ndarray:
    N = flat1.shape[0]
    result = np.empty((N, 6), dtype=flat1.dtype)
    for i in range(N):
        result[i] = _se3_triads2euler_sv(flat1[i], flat2[i])
    return result


@cond_jit(nopython=True,cache=True)
def _se3_midstep2triad_flat(flat: np.ndarray) -> np.ndarray:
    N = flat.shape[0]
    result = np.empty((N, 6), dtype=flat.dtype)
    for i in range(N):
        result[i] = _se3_midstep2triad_sv(flat[i])
    return result


@cond_jit(nopython=True,cache=True)
def _se3_triad2midstep_flat(flat: np.ndarray) -> np.ndarray:
    N = flat.shape[0]
    result = np.empty((N, 6), dtype=flat.dtype)
    for i in range(N):
        result[i] = _se3_triad2midstep_sv(flat[i])
    return result


@cond_jit(nopython=True,cache=True)
def _se3_triadxrotmat_midsteptrans_flat(flat1: np.ndarray, flat2: np.ndarray) -> np.ndarray:
    N = flat1.shape[0]
    result = np.empty((N, 4, 4), dtype=flat1.dtype)
    for i in range(N):
        result[i] = _se3_triadxrotmat_midsteptrans_sv(flat1[i], flat2[i])
    return result


@cond_jit(nopython=True,cache=True)
def _se3_triads2rotmat_midsteptrans_flat(flat1: np.ndarray, flat2: np.ndarray) -> np.ndarray:
    N = flat1.shape[0]
    result = np.empty((N, 4, 4), dtype=flat1.dtype)
    for i in range(N):
        result[i] = _se3_triads2rotmat_midsteptrans_sv(flat1[i], flat2[i])
    return result


@cond_jit(nopython=True,cache=True)
def _se3_transformation_triad2midstep_flat(flat: np.ndarray) -> np.ndarray:
    N = flat.shape[0]
    result = np.empty((N, 4, 4), dtype=flat.dtype)
    for i in range(N):
        result[i] = _se3_transformation_triad2midstep_sv(flat[i])
    return result


@cond_jit(nopython=True,cache=True)
def _se3_transformation_midstep2triad_flat(flat: np.ndarray) -> np.ndarray:
    N = flat.shape[0]
    result = np.empty((N, 4, 4), dtype=flat.dtype)
    for i in range(N):
        result[i] = _se3_transformation_midstep2triad_sv(flat[i])
    return result


@cond_jit(nopython=True,cache=True)
def _se3_algebra2group_lintrans_flat(flat: np.ndarray, translation_as_midstep: bool) -> np.ndarray:
    N = flat.shape[0]
    result = np.empty((N, 6, 6), dtype=flat.dtype)
    for i in range(N):
        result[i] = _se3_algebra2group_lintrans_sv(flat[i], translation_as_midstep)
    return result


@cond_jit(nopython=True,cache=True)
def _se3_group2algebra_lintrans_flat(flat: np.ndarray, translation_as_midstep: bool) -> np.ndarray:
    N = flat.shape[0]
    result = np.empty((N, 6, 6), dtype=flat.dtype)
    for i in range(N):
        result[i] = _se3_group2algebra_lintrans_sv(flat[i], translation_as_midstep)
    return result


@cond_jit(nopython=True,cache=True)
def _se3_algebra2group_stiffmat_flat(
    flat_gs: np.ndarray, flat_stiff: np.ndarray, translation_as_midstep: bool
) -> np.ndarray:
    N = flat_gs.shape[0]
    result = np.empty((N, 6, 6), dtype=flat_gs.dtype)
    for i in range(N):
        result[i] = _se3_algebra2group_stiffmat_sv(flat_gs[i], flat_stiff[i], translation_as_midstep)
    return result


@cond_jit(nopython=True,cache=True)
def _se3_group2algebra_stiffmat_flat(
    flat_gs: np.ndarray, flat_stiff: np.ndarray, translation_as_midstep: bool
) -> np.ndarray:
    N = flat_gs.shape[0]
    result = np.empty((N, 6, 6), dtype=flat_gs.dtype)
    for i in range(N):
        result[i] = _se3_group2algebra_stiffmat_sv(flat_gs[i], flat_stiff[i], translation_as_midstep)
    return result


##############################################################################
# Layer 3 – Python dispatchers (original public names)
##############################################################################

def se3_inverse_batch(g: np.ndarray) -> np.ndarray:
    """Inverse of element(s) of SE3.

    Parameters
    ----------
    g : (..., 4, 4)  SE3 element(s)

    Returns
    -------
    inv : (..., 4, 4)
    """
    g = np.asarray(g, dtype=float)
    if g.ndim == 2:
        return _se3_inverse_sv(g)
    shape = g.shape[:-2]
    n = int(np.prod(np.array(shape)))
    flat = np.ascontiguousarray(g.reshape(n, 4, 4))
    return _se3_inverse_flat(flat).reshape(shape + (4, 4))


def se3_triads2rotmat_batch(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """SE3 transformation matrix mapping tau1 into tau2 w.r.t. frame of tau1.

    Parameters
    ----------
    tau1, tau2 : (..., 4, 4)

    Returns
    -------
    g : (..., 4, 4)
    """
    tau1 = np.asarray(tau1, dtype=float)
    tau2 = np.asarray(tau2, dtype=float)
    if tau1.ndim == 2:
        return _se3_triads2rotmat_sv(tau1, tau2)
    shape = tau1.shape[:-2]
    n = int(np.prod(np.array(shape)))
    flat1 = np.ascontiguousarray(tau1.reshape(n, 4, 4))
    flat2 = np.ascontiguousarray(tau2.reshape(n, 4, 4))
    return _se3_triads2rotmat_flat(flat1, flat2).reshape(shape + (4, 4))


def se3_triads2euler_batch(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """SE3 Euler vector mapping tau1 into tau2.

    Parameters
    ----------
    tau1, tau2 : (..., 4, 4)

    Returns
    -------
    euler : (..., 6)
    """
    tau1 = np.asarray(tau1, dtype=float)
    tau2 = np.asarray(tau2, dtype=float)
    if tau1.ndim == 2:
        return _se3_triads2euler_sv(tau1, tau2)
    shape = tau1.shape[:-2]
    n = int(np.prod(np.array(shape)))
    flat1 = np.ascontiguousarray(tau1.reshape(n, 4, 4))
    flat2 = np.ascontiguousarray(tau2.reshape(n, 4, 4))
    return _se3_triads2euler_flat(flat1, flat2).reshape(shape + (6,))


def se3_midstep2triad_batch(triad_euler: np.ndarray) -> np.ndarray:
    """Convert SE3 Euler vector from midstep to triad convention.

    Parameters
    ----------
    triad_euler : (..., 6)

    Returns
    -------
    midstep_euler : (..., 6)
    """
    triad_euler = np.asarray(triad_euler, dtype=float)
    if triad_euler.ndim == 1:
        return _se3_midstep2triad_sv(triad_euler)
    shape = triad_euler.shape[:-1]
    n = int(np.prod(np.array(shape)))
    flat = np.ascontiguousarray(triad_euler.reshape(n, 6))
    return _se3_midstep2triad_flat(flat).reshape(shape + (6,))


def se3_triad2midstep_batch(midstep_euler: np.ndarray) -> np.ndarray:
    """Convert SE3 Euler vector from triad to midstep convention.

    Parameters
    ----------
    midstep_euler : (..., 6)

    Returns
    -------
    triad_euler : (..., 6)
    """
    midstep_euler = np.asarray(midstep_euler, dtype=float)
    if midstep_euler.ndim == 1:
        return _se3_triad2midstep_sv(midstep_euler)
    shape = midstep_euler.shape[:-1]
    n = int(np.prod(np.array(shape)))
    flat = np.ascontiguousarray(midstep_euler.reshape(n, 6))
    return _se3_triad2midstep_flat(flat).reshape(shape + (6,))


def se3_triadxrotmat_midsteptrans_batch(tau1: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Triad × SE3 rotation matrix (translation in midstep frame).

    Parameters
    ----------
    tau1, g : (..., 4, 4)

    Returns
    -------
    tau2 : (..., 4, 4)
    """
    tau1 = np.asarray(tau1, dtype=float)
    g = np.asarray(g, dtype=float)
    if tau1.ndim == 2:
        return _se3_triadxrotmat_midsteptrans_sv(tau1, g)
    shape = tau1.shape[:-2]
    n = int(np.prod(np.array(shape)))
    flat1 = np.ascontiguousarray(tau1.reshape(n, 4, 4))
    flatg = np.ascontiguousarray(g.reshape(n, 4, 4))
    return _se3_triadxrotmat_midsteptrans_flat(flat1, flatg).reshape(shape + (4, 4))


def se3_triads2rotmat_midsteptrans_batch(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """SE3 transform from tau1 to tau2 with midstep translation convention.

    Parameters
    ----------
    tau1, tau2 : (..., 4, 4)

    Returns
    -------
    g : (..., 4, 4)
    """
    tau1 = np.asarray(tau1, dtype=float)
    tau2 = np.asarray(tau2, dtype=float)
    if tau1.ndim == 2:
        return _se3_triads2rotmat_midsteptrans_sv(tau1, tau2)
    shape = tau1.shape[:-2]
    n = int(np.prod(np.array(shape)))
    flat1 = np.ascontiguousarray(tau1.reshape(n, 4, 4))
    flat2 = np.ascontiguousarray(tau2.reshape(n, 4, 4))
    return _se3_triads2rotmat_midsteptrans_flat(flat1, flat2).reshape(shape + (4, 4))


def se3_transformation_triad2midstep_batch(g: np.ndarray) -> np.ndarray:
    """Transform translation of SE3 element from triad to midstep convention.

    Parameters
    ----------
    g : (..., 4, 4)

    Returns
    -------
    midg : (..., 4, 4)
    """
    g = np.asarray(g, dtype=float)
    if g.ndim == 2:
        return _se3_transformation_triad2midstep_sv(g)
    shape = g.shape[:-2]
    n = int(np.prod(np.array(shape)))
    flat = np.ascontiguousarray(g.reshape(n, 4, 4))
    return _se3_transformation_triad2midstep_flat(flat).reshape(shape + (4, 4))


def se3_transformation_midstep2triad_batch(midg: np.ndarray) -> np.ndarray:
    """Transform translation of SE3 element from midstep to triad convention.

    Parameters
    ----------
    midg : (..., 4, 4)

    Returns
    -------
    g : (..., 4, 4)
    """
    midg = np.asarray(midg, dtype=float)
    if midg.ndim == 2:
        return _se3_transformation_midstep2triad_sv(midg)
    shape = midg.shape[:-2]
    n = int(np.prod(np.array(shape)))
    flat = np.ascontiguousarray(midg.reshape(n, 4, 4))
    return _se3_transformation_midstep2triad_flat(flat).reshape(shape + (4, 4))


def se3_algebra2group_lintrans_batch(
    groundstate_algebra: np.ndarray, translation_as_midstep: bool = False
) -> np.ndarray:
    """Linear transform from algebra to group splitting at a given groundstate.

    Parameters
    ----------
    groundstate_algebra : (..., 6)
    translation_as_midstep : bool

    Returns
    -------
    Trans : (..., 6, 6)
    """
    groundstate_algebra = np.asarray(groundstate_algebra, dtype=float)
    if groundstate_algebra.ndim == 1:
        return _se3_algebra2group_lintrans_sv(groundstate_algebra, translation_as_midstep)
    shape = groundstate_algebra.shape[:-1]
    n = int(np.prod(np.array(shape)))
    flat = np.ascontiguousarray(groundstate_algebra.reshape(n, 6))
    return _se3_algebra2group_lintrans_flat(flat, translation_as_midstep).reshape(shape + (6, 6))


def se3_group2algebra_lintrans_batch(
    groundstate_group: np.ndarray, translation_as_midstep: bool = False
) -> np.ndarray:
    """Linear transform from group to algebra splitting at a given groundstate.

    Parameters
    ----------
    groundstate_group : (..., 6)
    translation_as_midstep : bool

    Returns
    -------
    Trans : (..., 6, 6)
    """
    groundstate_group = np.asarray(groundstate_group, dtype=float)
    if groundstate_group.ndim == 1:
        return _se3_group2algebra_lintrans_sv(groundstate_group, translation_as_midstep)
    shape = groundstate_group.shape[:-1]
    n = int(np.prod(np.array(shape)))
    flat = np.ascontiguousarray(groundstate_group.reshape(n, 6))
    return _se3_group2algebra_lintrans_flat(flat, translation_as_midstep).reshape(shape + (6, 6))


def se3_algebra2group_stiffmat_batch(
    groundstate_algebra: np.ndarray,
    stiff_algebra: np.ndarray,
    translation_as_midstep: bool = False,
) -> np.ndarray:
    """Stiffness matrix from algebra to group representation.

    Parameters
    ----------
    groundstate_algebra : (..., 6)
    stiff_algebra : (..., 6, 6)
    translation_as_midstep : bool

    Returns
    -------
    stiff_group : (..., 6, 6)
    """
    groundstate_algebra = np.asarray(groundstate_algebra, dtype=float)
    stiff_algebra = np.asarray(stiff_algebra, dtype=float)
    if groundstate_algebra.ndim == 1:
        return _se3_algebra2group_stiffmat_sv(groundstate_algebra, stiff_algebra, translation_as_midstep)
    shape = groundstate_algebra.shape[:-1]
    n = int(np.prod(np.array(shape)))
    flat_gs = np.ascontiguousarray(groundstate_algebra.reshape(n, 6))
    flat_stiff = np.ascontiguousarray(stiff_algebra.reshape(n, 6, 6))
    return _se3_algebra2group_stiffmat_flat(flat_gs, flat_stiff, translation_as_midstep).reshape(shape + (6, 6))


def se3_group2algebra_stiffmat_batch(
    groundstate_group: np.ndarray,
    stiff_group: np.ndarray,
    translation_as_midstep: bool = False,
) -> np.ndarray:
    """Stiffness matrix from group to algebra representation.

    Parameters
    ----------
    groundstate_group : (..., 6)
    stiff_group : (..., 6, 6)
    translation_as_midstep : bool

    Returns
    -------
    stiff_algebra : (..., 6, 6)
    """
    groundstate_group = np.asarray(groundstate_group, dtype=float)
    stiff_group = np.asarray(stiff_group, dtype=float)
    if groundstate_group.ndim == 1:
        return _se3_group2algebra_stiffmat_sv(groundstate_group, stiff_group, translation_as_midstep)
    shape = groundstate_group.shape[:-1]
    n = int(np.prod(np.array(shape)))
    flat_gs = np.ascontiguousarray(groundstate_group.reshape(n, 6))
    flat_stiff = np.ascontiguousarray(stiff_group.reshape(n, 6, 6))
    return _se3_group2algebra_stiffmat_flat(flat_gs, flat_stiff, translation_as_midstep).reshape(shape + (6, 6))


##############################################################################
# Original single-input implementations (kept verbatim, renamed _single)
##############################################################################


@cond_jit(nopython=True,cache=True)
def se3_inverse_single(g: np.ndarray) -> np.ndarray:
    """Inverse of element of SE3"""
    inv = np.empty(g.shape, dtype=g.dtype)
    inv[:3, :3] = g[:3, :3].T
    inv[:3, 3] = -inv[:3, :3] @ g[:3, 3]
    inv[3, :]  = np.array([0, 0, 0, 1])
    return inv


@cond_jit(nopython=True,cache=True)
def se3_triads2rotmat_single(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """find SE3 transformation matrix, g, that maps tau1 into tau2 with respect to the frame of tau1"""
    return se3_inverse_single(tau1) @ tau2


@cond_jit(nopython=True,cache=True)
def se3_triads2euler_single(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    return se3_rotmat2euler_single(se3_triads2rotmat_single(tau1, tau2))


@cond_jit(nopython=True,cache=True)
def se3_midstep2triad_single(midstep_euler: np.ndarray) -> np.ndarray:
    triad_euler = np.copy(midstep_euler)
    vrot = midstep_euler[:3]
    vtrans = midstep_euler[3:]
    sqrt_rotmat = euler2rotmat_single(0.5 * vrot)
    triad_euler[3:] = sqrt_rotmat @ vtrans
    return triad_euler


@cond_jit(nopython=True,cache=True)
def se3_triad2midstep_single(triad_euler: np.ndarray) -> np.ndarray:
    midstep_euler = np.copy(triad_euler)
    vrot = triad_euler[:3]
    vtrans = triad_euler[3:]
    sqrt_rotmat = euler2rotmat_single(0.5 * vrot)
    midstep_euler[3:] = sqrt_rotmat.T @ vtrans
    return midstep_euler


@cond_jit(nopython=True,cache=True)
def se3_triadxrotmat_midsteptrans_single(tau1: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Multiplication of triad with rotation matrix g (in SE3) assuming that the translation of g is defined with respect to the midstep triad."""
    R = g[:3, :3]
    T1 = tau1[:3, :3]
    tau2 = np.eye(4)
    tau2[:3, :3] = T1 @ R
    tau2[:3, 3] = tau1[:3, 3] + T1 @ sqrt_rot_single(R) @ g[:3, 3]
    return tau2


@cond_jit(nopython=True,cache=True)
def se3_triads2rotmat_midsteptrans_single(tau1: np.ndarray, tau2: np.ndarray) -> np.ndarray:
    """find SE3 transformation matrix, g, that maps tau1 into tau2 with respect to the frame of tau1, assuming that the translation of g is defined with respect to the midstep triad."""
    T1 = tau1[:3, :3]
    T2 = tau2[:3, :3]
    R = T1.T @ T2
    Tmid = T1 @ sqrt_rot_single(R)
    zeta = Tmid.T @ (tau2[:3, 3] - tau1[:3, 3])
    g = np.eye(4)
    g[:3, :3] = R
    g[:3, 3] = zeta
    return g


@cond_jit(nopython=True,cache=True)
def se3_transformation_triad2midstep_single(g: np.ndarray) -> np.ndarray:
    """transforms translation of transformation g (in SE3) from canonical definition to mid-step triad definition."""
    midg = np.copy(g)
    midg[:3, 3] = np.transpose(sqrt_rot_single(g[:3, :3])) @ g[:3, 3]
    return midg


@cond_jit(nopython=True,cache=True)
def se3_transformation_midstep2triad_single(midg: np.ndarray) -> np.ndarray:
    """transforms translation of transformation g (in SE3) from mid-step triad definition to canonical definition."""
    g = np.copy(midg)
    g[:3, 3] = sqrt_rot_single(midg[:3, :3]) @ midg[:3, 3]
    return g


@cond_jit(nopython=True,cache=True)
def se3_algebra2group_lintrans_single(
    groundstate_algebra: np.ndarray, translation_as_midstep: bool = False
) -> np.ndarray:
    Trans = np.eye(6)
    Omega_0 = groundstate_algebra[:3]
    zeta_0 = groundstate_algebra[3:]

    Trans[:3, :3] = splittransform_algebra2group_single(Omega_0)
    if translation_as_midstep:
        sqrtS_transp = euler2rotmat_single(-0.5 * Omega_0)
        zeta_0_hat_transp = hat_map_single(-zeta_0)
        H_half = splittransform_algebra2group_single(0.5 * Omega_0)
        Trans[3:, :3] = 0.5 * sqrtS_transp @ zeta_0_hat_transp @ H_half
        Trans[3:, 3:] = sqrtS_transp
    else:
        Trans[3:, 3:] = euler2rotmat_single(-Omega_0)
    return Trans


@cond_jit(nopython=True,cache=True)
def se3_group2algebra_lintrans_single(
    groundstate_group: np.ndarray, translation_as_midstep: bool = False
) -> np.ndarray:
    Trans = np.eye(6)
    Phi_0 = groundstate_group[:3]
    s = groundstate_group[3:]

    H_inv = splittransform_group2algebra_single(Phi_0)
    Trans[:3, :3] = H_inv
    if translation_as_midstep:
        sqrtS = euler2rotmat_single(0.5 * Phi_0)
        zeta_0 = sqrtS.T @ s
        zeta_0_hat_transp = hat_map_single(-zeta_0)
        H_half = splittransform_algebra2group_single(0.5 * Phi_0)
        Trans[3:, :3] = -0.5 * zeta_0_hat_transp @ H_half @ H_inv
        Trans[3:, 3:] = sqrtS
    else:
        Trans[3:, 3:] = euler2rotmat_single(Phi_0)
    return Trans


@cond_jit(nopython=True,cache=True)
def se3_algebra2group_stiffmat_single(
    groundstate_algebra: np.ndarray,
    stiff_algebra: np.ndarray,
    translation_as_midstep: bool = False,
) -> np.ndarray:
    """Converts stiffness matrix from algebra-level (vector) splitting between static and dynamic component to group-level (matrix) splitting. Optionally, the transformations from midstep triad definition to triad definition of the translational component may also be included."""
    HX = se3_algebra2group_lintrans_single(
        groundstate_algebra, translation_as_midstep
    )
    HX_inv = np.linalg.inv(HX)
    stiff_group = HX_inv.T @ stiff_algebra @ HX_inv
    return stiff_group


@cond_jit(nopython=True,cache=True)
def se3_group2algebra_stiffmat_single(
    groundstate_group: np.ndarray,
    stiff_group: np.ndarray,
    translation_as_midstep: bool = False,
) -> np.ndarray:
    """Converts stiffness matrix from group-level (matrix) splitting between static and dynamic component to algebra-level (vector) splitting. Optionally, the transformations from midstep triad definition to triad definition of the translational component may also be included. I.e. the final
    definition will assume a midstep triad definition of the translational component.
    """
    HX_inv = se3_group2algebra_lintrans_single(
        groundstate_group, translation_as_midstep
    )
    HX = np.linalg.inv(HX_inv)
    stiff_algebra = HX.T @ stiff_group @ HX
    return stiff_algebra

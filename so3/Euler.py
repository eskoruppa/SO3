#!/bin/env python3

import numpy as np

from .pyConDec.pycondec import cond_jit


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

# @cond_jit(nopython=True, cache=True)
def euler2rotmat(Omega: np.ndarray) -> np.ndarray:
    """
    Euler-Rodrigues / exponential map from so(3) to SO(3).

    Numerically stable for small ||Omega|| by using series expansions for
    sin(x)/x and (1-cos(x))/x^2.
    """
    Om = np.linalg.norm(Omega)

    # Identity for (near) zero rotation
    if Om < DEF_EULER_EPSILON:
        return np.eye(3, dtype=np.double)

    # Stable evaluation of:
    #   A = sin(Om)/Om
    #   B = (1-cos(Om))/Om^2
    if Om < DEF_EULER_SERIES_SMALL:
        Om2 = Om * Om
        Om4 = Om2 * Om2
        # sin(x)/x = 1 - x^2/6 + x^4/120
        A = 1.0 - Om2 / 6.0 + Om4 / 120.0
        # (1-cos(x))/x^2 = 1/2 - x^2/24 + x^4/720
        B = 0.5 - Om2 / 24.0 + Om4 / 720.0
    else:
        A = np.sin(Om) / Om
        B = (1.0 - np.cos(Om)) / (Om * Om)

    x = Omega[0]
    y = Omega[1]
    z = Omega[2]

    # Rodrigues formula: R = I + A*[w]_x + B*[w]_x^2 (expanded)
    R = np.empty((3, 3), dtype=np.double)

    xx = x * x
    yy = y * y
    zz = z * z

    xy = x * y
    xz = x * z
    yz = y * z

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

# @cond_jit(nopython=True, cache=True)
def rotmat2euler(R: np.ndarray) -> np.ndarray:
    """
    Inverse of Euler Rodriguez Formula (log map SO(3)->so(3)),
    returning the rotation vector Omega (3,).

    Robustness features (as in the earlier version):
      - clamp trace-derived val into [-1, 1] to avoid arccos NaNs
      - stable small-angle handling for th/(2 sin th)
      - robust angle≈pi handling using "largest diagonal" axis extraction
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    val = 0.5 * (tr - 1.0)

    # Clamp to valid arccos domain
    if val > 1.0:
        val = 1.0
    elif val < -1.0:
        val = -1.0

    # angle ~ 0
    if val > DEF_EULER_CLOSE_TO_ONE:
        return np.zeros(3, dtype=np.double)

    # angle ~ pi
    if val < DEF_EULER_CLOSE_TO_MINUS_ONE:
        r00 = R[0, 0]
        r11 = R[1, 1]
        r22 = R[2, 2]

        # Compute axis components with safeguards against tiny negative due to roundoff
        if (r00 >= r11) and (r00 >= r22):
            t = 0.5 * (r00 + 1.0)
            if t < 0.0:
                t = 0.0
            ax = np.sqrt(t)
            if ax < DEF_AXIS_COMP_EPS:
                # fallback (rare): choose another axis deterministically
                t1 = 0.5 * (r11 + 1.0)
                if t1 < 0.0:
                    t1 = 0.0
                ay = np.sqrt(t1)
                t2 = 0.5 * (r22 + 1.0)
                if t2 < 0.0:
                    t2 = 0.0
                az = np.sqrt(t2)
                if ay >= az:
                    return np.array([0.0, np.pi, 0.0], dtype=np.double)
                return np.array([0.0, 0.0, np.pi], dtype=np.double)
            ay = R[0, 1] / (2.0 * ax)
            az = R[0, 2] / (2.0 * ax)

        elif r11 >= r22:
            t = 0.5 * (r11 + 1.0)
            if t < 0.0:
                t = 0.0
            ay = np.sqrt(t)
            if ay < DEF_AXIS_COMP_EPS:
                t0 = 0.5 * (r00 + 1.0)
                if t0 < 0.0:
                    t0 = 0.0
                ax = np.sqrt(t0)
                t2 = 0.5 * (r22 + 1.0)
                if t2 < 0.0:
                    t2 = 0.0
                az = np.sqrt(t2)
                if ax >= az:
                    return np.array([np.pi, 0.0, 0.0], dtype=np.double)
                return np.array([0.0, 0.0, np.pi], dtype=np.double)
            ax = R[0, 1] / (2.0 * ay)
            az = R[1, 2] / (2.0 * ay)

        else:
            t = 0.5 * (r22 + 1.0)
            if t < 0.0:
                t = 0.0
            az = np.sqrt(t)
            if az < DEF_AXIS_COMP_EPS:
                t0 = 0.5 * (r00 + 1.0)
                if t0 < 0.0:
                    t0 = 0.0
                ax = np.sqrt(t0)
                t1 = 0.5 * (r11 + 1.0)
                if t1 < 0.0:
                    t1 = 0.0
                ay = np.sqrt(t1)
                if ax >= ay:
                    return np.array([np.pi, 0.0, 0.0], dtype=np.double)
                return np.array([0.0, np.pi, 0.0], dtype=np.double)
            ax = R[0, 2] / (2.0 * az)
            ay = R[1, 2] / (2.0 * az)

        # Normalize axis, then scale by pi
        nrm = np.sqrt(ax * ax + ay * ay + az * az)
        if nrm < DEF_AXIS_NORM_EPS:
            return np.array([np.pi, 0.0, 0.0], dtype=np.double)
        inv = np.pi / nrm
        return np.array([ax * inv, ay * inv, az * inv], dtype=np.double)

    # general case
    th = np.arccos(val)

    # vee(R - R^T)
    vx = R[2, 1] - R[1, 2]
    vy = R[0, 2] - R[2, 0]
    vz = R[1, 0] - R[0, 1]

    # scale = th / (2 sin(th)) with stable small-angle approximation
    # th/(2 sin th) ~ 0.5 + th^2/12 for th -> 0
    if th < DEF_THETA_SCALE_SMALL:
        th2 = th * th
        scale = 0.5 + th2 / 12.0
    else:
        scale = 0.5 * th / np.sin(th)

    return np.array([scale * vx, scale * vy, scale * vz], dtype=np.double)



# DEF_EULER_EPSILON = 1e-12
# DEF_EULER_CLOSE_TO_ONE = 0.999999999999
# DEF_EULER_CLOSE_TO_MINUS_ONE = -0.999999999999

# @cond_jit(nopython=True,cache=True)
# def euler2rotmat(Omega: np.ndarray) -> np.ndarray:
#     """Returns the matrix version of the Euler-Rodrigues formula

#     Args:
#         Omega (np.ndarray): Euler vector / Rotation vector (3-vector)

#     Returns:
#         np.ndarray: Rotation matrix (element of SO(3))
#     """
#     Om = np.linalg.norm(Omega)
#     R = np.zeros((3, 3), dtype=np.double)

#     # if norm is zero, return identity matrix
#     if Om < DEF_EULER_EPSILON:
#         return np.eye(3)

#     cosOm = np.cos(Om)
#     sinOm = np.sin(Om)
#     Omsq = Om * Om
#     fac1 = (1 - cosOm) / Omsq
#     fac2 = sinOm / Om

#     R[0, 0] = cosOm + Omega[0] ** 2 * fac1
#     R[1, 1] = cosOm + Omega[1] ** 2 * fac1
#     R[2, 2] = cosOm + Omega[2] ** 2 * fac1
#     A = Omega[0] * Omega[1] * fac1
#     B = Omega[2] * fac2
#     R[0, 1] = A - B
#     R[1, 0] = A + B
#     A = Omega[0] * Omega[2] * fac1
#     B = Omega[1] * fac2
#     R[0, 2] = A + B
#     R[2, 0] = A - B
#     A = Omega[1] * Omega[2] * fac1
#     B = Omega[0] * fac2
#     R[1, 2] = A - B
#     R[2, 1] = A + B
#     return R


# # @cond_jit(nopython=True,cache=True)
# # def rotmat2euler(R: np.ndarray) -> np.ndarray:
# #     """Inversion of Euler Rodriguez Formula

# #     Args:
# #         R (np.ndarray): Rotation matrix (element of SO(3))

# #     Returns:
# #         np.ndarray: Euler vector / Rotation vector (3-vector)
# #     """
# #     val = 0.5 * (np.trace(R) - 1)
# #     if val > DEF_EULER_CLOSE_TO_ONE:
# #         return np.zeros(3)
# #     if val < DEF_EULER_CLOSE_TO_MINUS_ONE:
# #         if R[0, 0] > DEF_EULER_CLOSE_TO_ONE:
# #             return np.array([np.pi, 0, 0])
# #         if R[1, 1] > DEF_EULER_CLOSE_TO_ONE:
# #             return np.array([0, np.pi, 0])
# #         return np.array([0, 0, np.pi])
# #     Th = np.arccos(val)
# #     Theta = np.array([(R[2, 1] - R[1, 2]), (R[0, 2] - R[2, 0]), (R[1, 0] - R[0, 1])])
# #     Theta = Th * 0.5 / np.sin(Th) * Theta
# #     return Theta

# @cond_jit(nopython=True,cache=True)
# def rotmat2euler(R: np.ndarray) -> np.ndarray:
#     """Inversion of Euler Rodriguez Formula

#     Args:
#         R (np.ndarray): Rotation matrix (element of SO(3))

#     Returns:
#         np.ndarray: Euler vector / Rotation vector (3-vector)
#     """
#     val = 0.5 * (np.trace(R) - 1)
#     if val > DEF_EULER_CLOSE_TO_ONE:
#         return np.zeros(3)
#     if val < DEF_EULER_CLOSE_TO_MINUS_ONE:
#         # rotation around first axis by angle pi
#         if R[0, 0] > DEF_EULER_CLOSE_TO_ONE:
#             return np.array([np.pi, 0, 0])
#         # rotation around second axis by angle pi
#         if R[1, 1] > DEF_EULER_CLOSE_TO_ONE:
#             return np.array([0, np.pi, 0])
#         # rotation around third axis by angle pi
#         if R[2, 2] > DEF_EULER_CLOSE_TO_ONE:
#             return np.array([0, 0, np.pi])
#         # rotation around arbitrary axis by angle pi
#         A = R - np.eye(3)       
#         b = np.cross(A[0],A[1])
#         th = b - np.dot(b,A[2])*A[2]
#         th = th / np.linalg.norm(th) * np.pi
#         return th
#     Th = np.arccos(val)
#     Theta = np.array([(R[2, 1] - R[1, 2]), (R[0, 2] - R[2, 0]), (R[1, 0] - R[0, 1])])
#     Theta = Th * 0.5 / np.sin(Th) * Theta
#     return Theta


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

#!/bin/env python3
import numpy as np

def unwrap_euler(eulers: np.ndarray, copy: bool = True):
    """Unwrap a sequence of Euler (rotation) vectors across the 2π branch cut.

    A rotation vector ``Ω`` and ``(‖Ω‖ + 2πk)·Ω/‖Ω‖`` for integer ``k`` describe
    rotations that differ only by full turns about the same axis. Along a
    trajectory this can cause the rotation-vector magnitude to jump by ~2π
    between consecutive samples. This routine adjusts each vector by the whole
    multiple of 2π (along its own axis) that keeps it closest to its
    predecessor, yielding a continuous sequence — the rotation-vector analogue
    of :func:`numpy.unwrap` for phase angles.

    Args:
        eulers (np.ndarray): Sequence of Euler/rotation vectors, shape ``(N, 3)``.
        copy (bool): If True (default) operate on a copy and leave the input
            untouched; if False, modify ``eulers`` in place and return it.

    Returns:
        np.ndarray: The unwrapped sequence, same shape as the input.
    """
    if copy:
        exteulers = np.copy(eulers)
    else:
        exteulers = eulers
    for s in range(1,len(exteulers)):
        Om1 = exteulers[s-1]
        Om2 = exteulers[s]
        nOm2 = np.linalg.norm(Om2)
        uOm2 = Om2/nOm2
        shift = np.round (1./(2*np.pi) * (np.dot(Om1,uOm2) - nOm2))
        if np.abs(shift) > 0:
            nOm2p = nOm2 + shift * 2*np.pi
            Om2p = nOm2p * uOm2
            exteulers[s] = Om2p
    return exteulers


# Backwards-compatible alias for the original name. `unwrap_euler` is the
# canonical name (matches the numpy.unwrap convention); `extend_euler` is kept
# so existing code continues to work.
extend_euler = unwrap_euler

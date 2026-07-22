#!/bin/env python3
import numpy as np
from ._pycondec import cond_jit


@cond_jit(nopython=True, cache=True)
def _unwrap_euler_inplace(exteulers: np.ndarray) -> np.ndarray:
    """numba-jitted core: unwrap a float64 ``(B, N, 3)`` array of rotation
    vectors in place and return it. Each of the ``B`` batches is unwrapped
    independently along its ``N`` sequence axis. Assumes the caller has
    flattened leading dimensions and validated shape and dtype."""
    two_pi = 2.0 * np.pi
    n_batches = exteulers.shape[0]
    n_samples = exteulers.shape[1]
    if n_samples == 0:
        return exteulers

    for b in range(n_batches):
        prev = exteulers[b, 0]
        for s in range(1, n_samples):
            cur = exteulers[b, s]
            n = np.linalg.norm(cur)
            if n == 0.0:      # identity rotation: no axis, nothing to unwrap
                prev = cur
                continue
            axis = cur / n
            shift = np.round((np.dot(prev, axis) - n) / two_pi)
            if shift != 0.0:
                exteulers[b, s] = (n + shift * two_pi) * axis
            prev = exteulers[b, s]
    return exteulers


def unwrap_euler(eulers: np.ndarray, copy: bool = True):
    """Unwrap sequences of Euler (rotation) vectors across the 2π branch cut.

    A rotation vector ``Ω`` and ``(‖Ω‖ + 2πk)·Ω/‖Ω‖`` for integer ``k`` describe
    rotations that differ only by full turns about the same axis. Along a
    trajectory this can cause the rotation-vector magnitude to jump by ~2π
    between consecutive samples. This routine adjusts each vector by the whole
    multiple of 2π (along its own axis) that keeps it closest to its
    predecessor, yielding a continuous sequence — the rotation-vector analogue
    of :func:`numpy.unwrap` for phase angles.

    The last axis holds the 3-vector components and the second-to-last axis is
    the sequence that gets unwrapped. Any further leading axes — however many —
    are treated as independent batches. So ``(N, 3)``, ``(B, N, 3)``,
    ``(A, B, N, 3)`` and any ``(..., N, 3)`` are all accepted, each length-``N``
    sequence being unwrapped separately, and the result keeps the input shape.

    The input is always processed as a floating-point array, so integer inputs
    are handled without truncation.

    Args:
        eulers (np.ndarray): Euler/rotation vectors, shape ``(..., N, 3)`` with
            at least two dimensions (``(N, 3)`` for a single sequence).
        copy (bool): If True (default) operate on a copy and leave the input
            untouched; if False, modify ``eulers`` in place and return it (an
            integer or non-contiguous input is still copied, as it cannot hold
            the float results in place).

    Returns:
        np.ndarray: The unwrapped array as a float array, same shape as the input.

    Raises:
        ValueError: If ``eulers`` has fewer than two dimensions or its last axis
            is not of length 3 — e.g. a single Euler vector of shape ``(3,)``.
    """
    eulers = np.asarray(eulers)
    if eulers.ndim < 2 or eulers.shape[-1] != 3:
        raise ValueError(
            "unwrap_euler expects an array of Euler vectors with shape "
            f"(..., N, 3) and at least two dimensions, got array of shape "
            f"{eulers.shape}. To unwrap a single vector there is nothing to "
            "unwrap; pass a stacked sequence instead."
        )

    if copy:
        exteulers = np.array(eulers, dtype=np.double)
    else:
        exteulers = np.ascontiguousarray(eulers, dtype=np.double)

    n_samples = exteulers.shape[-2]
    n_batches = 1
    for d in exteulers.shape[:-2]:
        n_batches *= d
    batched = exteulers.reshape(n_batches, n_samples, 3)
    _unwrap_euler_inplace(batched)
    return exteulers

extend_euler = unwrap_euler

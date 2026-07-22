import numpy as np


def _check_vector_batch(x: np.ndarray, dim: int, name: str) -> None:
    if x.ndim < 1 or x.shape[-1] != dim:
        raise ValueError(
            f"{name} expects a length-{dim} vector or a batch of shape "
            f"(..., {dim}), got array of shape {x.shape}."
        )


def _check_matrix_batch(x: np.ndarray, m: int, name: str) -> None:
    if x.ndim < 2 or x.shape[-2:] != (m, m):
        raise ValueError(
            f"{name} expects a ({m}, {m}) matrix or a batch of shape "
            f"(..., {m}, {m}), got array of shape {x.shape}."
        )

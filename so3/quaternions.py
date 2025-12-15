import numpy as np
from .pyConDec.pycondec import cond_jit
import warnings

def depdec(repl=None):
    def decorator(func):
        msg = f"{func.__name__} is deprecated."
        if repl:
            msg += f" Use {repl} instead."

        def wrapper(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


@depdec(repl="quat2mat_numba")
@cond_jit(nopython=True,cache=True)
def _quat2rotmat(quat: np.ndarray) -> np.ndarray:
    """
    Converts a quaternion to a 3x3 rotation matrix, populating
    the columns with the rotated axis vectors. This function
    is a direct Python analog to the provided C++ `quat2rotmat`,
    meaning it does NOT normalize the input quaternion.

    The input quaternion `quat` is expected in the format [w, i, j, k],
    where 'w' is the scalar component and 'i, j, k' are the vector components.

    Args:
        quat (np.ndarray): A 1D NumPy array of 4 elements representing the quaternion [w, i, j, k].

    Returns:
        np.ndarray: A 3x3 NumPy array representing the rotation matrix.
                    The columns of this matrix are the rotated x, y, and z basis vectors.

    Raises:
        ValueError: If the input 'quat' is not a 1D NumPy array of 4 elements.
    """
    # if not isinstance(quat, np.ndarray) or quat.shape != (4,):
    #     print(quat.shape)
    #     raise ValueError("Input 'quat' must be a 1D NumPy array of 4 elements ([w, i, j, k]).")

    # Extract quaternion components
    w = quat[0]
    i = quat[1]
    j = quat[2]
    k = quat[3]

    # Calculate squared terms and products as in the C++ code
    w2 = w * w
    i2 = i * i
    j2 = j * j
    k2 = k * k
    twoij = 2.0 * i * j
    twoik = 2.0 * i * k
    twojk = 2.0 * j * k
    twoiw = 2.0 * i * w
    twojw = 2.0 * j * w
    twokw = 2.0 * k * w

    # Initialize the 3x3 matrix
    mat = np.empty((3, 3), dtype=float)

    # Populate the matrix elements based on the C++ logic
    # Note: C++ `mat[row][col]` directly maps to Python `mat[row, col]`
    # and we are essentially building a column-major matrix here.

    # First Column (Rotated X-axis vector)
    mat[0, 0] = w2 + i2 - j2 - k2
    mat[1, 0] = twoij - twokw
    mat[2, 0] = twojw + twoik

    # Second Column (Rotated Y-axis vector)
    mat[0, 1] = twoij + twokw
    mat[1, 1] = w2 - i2 + j2 - k2
    mat[2, 1] = twojk - twoiw

    # Third Column (Rotated Z-axis vector)
    mat[0, 2] = twoik - twojw
    mat[1, 2] = twojk + twoiw
    mat[2, 2] = w2 - i2 - j2 + k2

    return mat.T

@depdec(repl="quats2mats_numba")
@cond_jit(nopython=True,cache=True)
def quats2rotmats(quaternions_array: np.ndarray) -> np.ndarray:
    """
    Converts a NumPy array of quaternions (num_snapshots x num_atoms x 4)
    into a NumPy array of 3x3 rotation matrices (num_snapshots x num_atoms x 3 x 3).

    Each quaternion [w, i, j, k] is converted to a 3x3 rotation matrix
    where the columns represent the rotated x, y, and z basis vectors,
    analogous to the provided C++ `quat2rotmat` function.

    Args:
        quaternions_array (np.ndarray): A 3D NumPy array of shape (N, A, 4), where
                                         N = number of snapshots/timesteps
                                         A = number of atoms/rigid bodies
                                         4 = quaternion components [w, i, j, k]

    Returns:
        np.ndarray: A 4D NumPy array of shape (N, A, 3, 3), where
                    N = number of snapshots/timesteps
                    A = number of atoms/rigid bodies
                    3x3 = the rotation matrix (triad) for each rigid body.

    Raises:
        ValueError: If the input array does not have the expected shape (N, A, 4).
    """
    # if not isinstance(quaternions_array, np.ndarray) or quaternions_array.ndim != 3 or quaternions_array.shape[2] != 4:
    #     raise ValueError(
    #         "Input 'quaternions_array' must be a 3D NumPy array of shape (num_snapshots, num_atoms, 4)."
    #     )

    num_snapshots, num_atoms, _ = quaternions_array.shape

    # Pre-allocate the output array for efficiency
    triads_array = np.empty((num_snapshots, num_atoms, 3, 3), dtype=float)

    # Loop through each snapshot and each atom, converting its quaternion
    for i in range(num_snapshots):
        for j in range(num_atoms):
            quat = quaternions_array[i, j, :]
            triads_array[i, j, :, :] = _quat2rotmat(quat)

    return triads_array



@cond_jit(nopython=True, cache=True)
def quat2mat_numba(quat):
    """Numba core: quaternion (4,) -> rotation matrix (3,3), row-major like original."""
    w = quat[0]
    i = quat[1]
    j = quat[2]
    k = quat[3]

    w2 = w * w
    i2 = i * i
    j2 = j * j
    k2 = k * k
    twoij = 2.0 * i * j
    twoik = 2.0 * i * k
    twojk = 2.0 * j * k
    twoiw = 2.0 * i * w
    twojw = 2.0 * j * w
    twokw = 2.0 * k * w

    mat = np.empty((3, 3), dtype=np.float64)

    # row 0
    mat[0, 0] = w2 + i2 - j2 - k2
    mat[0, 1] = twoij - twokw
    mat[0, 2] = twojw + twoik

    # row 1
    mat[1, 0] = twoij + twokw
    mat[1, 1] = w2 - i2 + j2 - k2
    mat[1, 2] = twojk - twoiw

    # row 2
    mat[2, 0] = twoik - twojw
    mat[2, 1] = twojk + twoiw
    mat[2, 2] = w2 - i2 - j2 + k2

    return mat


@cond_jit(nopython=True, cache=True)
def mat2quat_numba(mat):
    """Numba core: rotation matrix (3,3) -> quaternion (4,) [w, x, y, z]."""
    m00 = mat[0, 0]
    m01 = mat[0, 1]
    m02 = mat[0, 2]
    m10 = mat[1, 0]
    m11 = mat[1, 1]
    m12 = mat[1, 2]
    m20 = mat[2, 0]
    m21 = mat[2, 1]
    m22 = mat[2, 2]

    tr = m00 + m11 + m22

    q0sq = 0.25 * (tr + 1.0)
    q1sq = q0sq - 0.5 * (m11 + m22)
    q2sq = q0sq - 0.5 * (m00 + m22)
    q3sq = q0sq - 0.5 * (m00 + m11)

    q0 = 0.0
    q1 = 0.0
    q2 = 0.0
    q3 = 0.0

    if q0sq >= 0.25:
        q0 = np.sqrt(q0sq)
        denom = 1.0 / (4.0 * q0)
        q1 = (m21 - m12) * denom
        q2 = (m02 - m20) * denom
        q3 = (m10 - m01) * denom
    elif q1sq >= 0.25:
        q1 = np.sqrt(q1sq)
        denom = 1.0 / (4.0 * q1)
        q0 = (m21 - m12) * denom
        q2 = (m01 + m10) * denom
        q3 = (m20 + m02) * denom
    elif q2sq >= 0.25:
        q2 = np.sqrt(q2sq)
        denom = 1.0 / (4.0 * q2)
        q0 = (m02 - m20) * denom
        q1 = (m01 + m10) * denom
        q3 = (m12 + m21) * denom
    else:
        q3 = np.sqrt(q3sq)
        denom = 1.0 / (4.0 * q3)
        q0 = (m10 - m01) * denom
        q1 = (m02 + m20) * denom
        q2 = (m12 + m21) * denom

    q = np.empty(4, dtype=np.float64)
    q[0] = q0
    q[1] = q1
    q[2] = q2
    q[3] = q3

    norm = np.sqrt(q0*q0 + q1*q1 + q2*q2 + q3*q3)
    if norm > 0.0:
        inv = 1.0 / norm
        q[0] *= inv
        q[1] *= inv
        q[2] *= inv
        q[3] *= inv

    return q


def quat2mat(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("Input 'quat' must be a 1D array-like of length 4.")
    return quat2mat_numba(q)


def mat2quat(mat) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError("Input 'mat' must have shape (3, 3).")
    return mat2quat_numba(m)


def quats2mats(quats: np.ndarray) -> np.ndarray:
    if len(quats.shape) == 1:
        return quat2mat_numba(quats)

    matshape = tuple(list(quats.shape)[:-1] + [3,3])
    mats = np.zeros((matshape),dtype=np.float64)
    if len(quats.shape) > 2:
        for i in range(len(quats)):
            mats[i] = quats2mats(quats[i])
        return mats
    for i in range(len(quats)):
       mats[i] = quat2mat_numba(quats[i])
    return mats


def mats2quats(mats: np.ndarray) -> np.ndarray:
    if len(mats.shape) == 2:
        return mat2quat_numba(mats)

    quatshape = tuple(list(mats.shape)[:-2] + [4])
    quats = np.zeros((quatshape),dtype=np.float64)
    if len(mats.shape) > 3:
        for i in range(len(mats)):
            quats[i] = mats2quats(mats[i])
        return quats
    for i in range(len(mats)):
       quats[i] = mat2quat_numba(mats[i])
    return quats


if __name__ == '__main__':
    
    from .Euler import euler2rotmat
    import sys
    
    dims = (5,6)
    
    quats = np.zeros(tuple(list(dims)+[4]))
    for i in range(dims[0]):
        for j in range(dims[1]):
            Om = np.random.uniform(-np.pi,np.pi,3)
            R = euler2rotmat(Om)
            quats[i,j] = mat2quat(R)
    
    T = quats2mats(quats)
    
    qs = mats2quats(T)
    
    print(np.sum(qs-quats))
    
    Om = np.random.uniform(-np.pi,np.pi,3)
    R = euler2rotmat(Om)
    q = mat2quat(R)
    
    print(q)
    print(q.shape)
    
    R1 = quat2mat(q)
    R2 = _quat2rotmat(q)
    
    print(R1-R2)
    print(R-R2)



import numpy as np

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
    if not isinstance(quat, np.ndarray) or quat.shape != (4,):
        print(quat.shape)
        raise ValueError("Input 'quat' must be a 1D NumPy array of 4 elements ([w, i, j, k]).")

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
    if not isinstance(quaternions_array, np.ndarray) or quaternions_array.ndim != 3 or quaternions_array.shape[2] != 4:
        raise ValueError(
            "Input 'quaternions_array' must be a 3D NumPy array of shape (num_snapshots, num_atoms, 4)."
        )

    num_snapshots, num_atoms, _ = quaternions_array.shape

    # Pre-allocate the output array for efficiency
    triads_array = np.empty((num_snapshots, num_atoms, 3, 3), dtype=float)

    # Loop through each snapshot and each atom, converting its quaternion
    for i in range(num_snapshots):
        for j in range(num_atoms):
            quat = quaternions_array[i, j, :]
            triads_array[i, j, :, :] = _quat2rotmat(quat)

    return triads_array


def quat2mat(quat):
    """
    Convert a quaternion to a 3x3 rotation matrix.

    Parameters
    ----------
    quat : sequence of float
        The quaternion [w, i, j, k].

    Returns
    -------
    mat : list of list of float
        The corresponding 3x3 rotation matrix.
    """
    w, i, j, k = quat

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

    mat = [
        [w2 + i2 - j2 - k2, twoij - twokw,     twojw + twoik    ],
        [twoij + twokw,     w2 - i2 + j2 - k2, twojk - twoiw    ],
        [twoik - twojw,     twojk + twoiw,     w2 - i2 - j2 + k2]
    ]

    return mat


def mat2quat(mat):
    """
    Convert a 3×3 rotation matrix to a quaternion [w, x, y, z].
    
    Parameters
    ----------
    mat : sequence of sequence of float
        3×3 rotation matrix, mat[row][col].
    
    Returns
    -------
    q : list of float
        Quaternion [w, x, y, z].
    """
    np.trace(mat)
    # Compute the “squares” terms
    q0sq = 0.25 * (np.trace(mat) + 1.0)
    q1sq = q0sq - 0.5 * (mat[1,1] + mat[2,2])
    q2sq = q0sq - 0.5 * (mat[0,0] + mat[2,2])
    q3sq = q0sq - 0.5 * (mat[0,0] + mat[1,1])

    # Prepare quaternion array
    q = np.zeros(4,dtype=np.double)

    # Choose the largest component to ensure numerical stability
    if q0sq >= 0.25:
        q[0] = np.sqrt(q0sq)
        denom = 1. / (4.0 * q[0])
        q[1] = (mat[2,1] - mat[1,2]) * denom
        q[2] = (mat[0,2] - mat[2,0]) * denom
        q[3] = (mat[1,0] - mat[0,1]) * denom
    elif q1sq >= 0.25:
        q[1] = np.sqrt(q1sq)
        denom = 1. / (4.0 * q[1])
        q[0] = (mat[2,1] - mat[1,2]) * denom
        q[2] = (mat[0,1] + mat[1,0]) * denom
        q[3] = (mat[2,0] + mat[0,2]) * denom
    elif q2sq >= 0.25:
        q[2] = np.sqrt(q2sq)
        denom = 1. / (4.0 * q[2])
        q[0] = (mat[0,2] - mat[2,0]) * denom
        q[1] = (mat[0,1] + mat[1,0]) * denom
        q[3] = (mat[1,2] + mat[2,1]) * denom
    else:
        q[3] = np.sqrt(q3sq)
        denom = 1. / (4.0 * q[3])
        q[0] = (mat[1,0] - mat[0,1]) * denom
        q[1] = (mat[0,2] + mat[2,0]) * denom
        q[2] = (mat[1,2] + mat[2,1]) * denom

    # Normalize to unit length
    norm = np.linalg.norm(q)
    if norm > 0.0:
        q = q / norm
    return q



if __name__ == '__main__':
    
    from .Euler import euler2rotmat
    import sys
    
    Om = np.random.uniform(-np.pi,np.pi,3)
    
    R = euler2rotmat(Om)
    q = mat2quat(R)
    
    print(q)
    print(q.shape)
    
    R1 = quat2mat(q)
    R2 = _quat2rotmat(q)
    
    print(R1-R2)
# SO3

**SO3** is a Python library providing a comprehensive set of mathematical tools for working with the rotation group SO(3) and the special Euclidean group SE(3).

The library supports multiple parameterizations of rotations—Euler (rotation) vectors, Cayley vectors, and quaternions—and provides consistent conversion routines between them. All computationally intensive functions can be transparently accelerated with [Numba](https://numba.pydata.org/) JIT compilation via the [pyConDec](https://github.com/eskoruppa/pyConDec) conditional-decorator utility; if Numba is not installed the functions run as plain NumPy code with no changes required. 

---

## Installation

> **The package is distributed on PyPI as `so3tools` and can be imported as either `so3` or `so3tools`**—i.e. `pip install so3tools`, then `import so3` *or* `import so3tools` (both names refer to the same package).

> 💡 **Installing [Numba](https://numba.pydata.org/) is strongly recommended.** SO3 works without it, but all numerically intensive routines run as plain (slower) NumPy code in that case. With Numba installed they are JIT-compiled to native machine code, typically giving a large speed-up on the hot paths. Every option below shows how to add it.

### Option 1 — Install from PyPI (recommended)

```bash
pip install so3tools
```

This automatically pulls in all required dependencies (`numpy`, `scipy`, and `pycondec`). To include the recommended Numba acceleration, install the `numba` extra:

```bash
pip install "so3tools[numba]"
```

(equivalently, `pip install so3tools numba`).

### Option 2 — Recursive clone + local install

Clone the repository with its `pyConDec` submodule and install locally. This is the simplest source-based workflow and needs no separate pyConDec step—the bundled submodule in `so3/pyConDec/` is picked up automatically:

```bash
git clone --recurse-submodules -j8 https://github.com/eskoruppa/SO3.git
cd SO3
pip install .
```

Then add the recommended Numba acceleration:

```bash
pip install numba          # or:  pip install ".[numba]"
```

### Option 3 — Install pyConDec separately, then clone + install SO3

Useful if you did a plain (non-recursive) clone, or want pyConDec managed independently. First install pyConDec—either from PyPI or from source:

```bash
# from PyPI (easiest)
pip install pycondec

# …or from source
git clone https://github.com/eskoruppa/pyConDec.git
cd pyConDec && pip install . && cd ..
```

Then clone and install SO3:

```bash
git clone https://github.com/eskoruppa/SO3.git
cd SO3
pip install .
```

And add the recommended Numba acceleration:

```bash
pip install numba
```

### Dependencies

| Dependency | Required | Notes |
|------------|----------|-------|
| `numpy` | **Yes** | Installed automatically by pip |
| `scipy` | **Yes** | Installed automatically by pip |
| `pycondec` | **Yes** | Conditional-decorator utility; installed automatically from PyPI (Options 1), or provided via the git submodule / separate install (Options 2–3) |
| `numba` | Recommended | Enables JIT acceleration of all hot-path functions; strongly recommended for performance |

---

## Functionality overview

### Euler vector (rotation-vector) parameterization — SO(3)
- **Exponential map** `euler2rotmat`: Euler vector → rotation matrix (Rodrigues' formula with series expansion near zero)
- **Logarithmic map** `rotmat2euler`: rotation matrix → Euler vector
- **Square-root rotation** `sqrt_rot`: compute the half-angle rotation
- **Jacobians**: right and left Jacobians of the exponential map and their inverses (`right_jacobian`, `left_jacobian`, `inverse_right_jacobian`, `inverse_left_jacobian`)
- **Midstep frame** `midstep`: midstep Euler vector between two frames
- **Euler-vector unwrapping** `unwrap_euler`: unwrap a sequence of Euler vectors to remove 2π branch jumps

### Cayley map parameterization — SO(3)
- `cayley2rotmat` / `rotmat2cayley`: convert between Cayley vectors and rotation matrices
- `cayley2euler` / `euler2cayley`: convert between Cayley and Euler vectors
- Linear-expansion and factor variants of the above conversions for use in tangent-space calculations

### Quaternion parameterization — SO(3)
- `quat2mat` / `mat2quat`: single quaternion ↔ rotation matrix
- `quats2mats` / `mats2quats`: batch conversion for arrays of quaternions / rotation matrices

### SE(3) — rigid-body transformations
- `se3_euler2rotmat` / `se3_rotmat2euler`: SE(3) Lie algebra ↔ 4×4 homogeneous matrices
- `se3_inverse`: fast matrix inverse for SE(3) elements
- Triad-based step parameterization: compute SE(3) transformations from pairs of body frames (`se3_triads2rotmat`, `se3_triads2euler`)
- Midstep-frame conversions between triad and midstep Euler representations (`se3_triad2midstep`, `se3_midstep2triad`)
- Stiffness / compliance matrix transformations between algebra and group frames (`se3_algebra2group_stiffmat`, `se3_group2algebra_stiffmat`, and corresponding linear-transform variants)
- Batch SE(3) utilities: `se3_eulers2rotmats`, `se3_rotmats2eulers`

### Lie algebra utilities
- **Hat map** `hat_map` / **vec map** `vec_map`: isomorphism between R³ and so(3) skew-symmetric matrices
- **Generators** `generator1`, `generator2`, `generator3`: basis elements of so(3)

### Rotation utilities
- `rotmat_align_vector`: compute the rotation matrix that maps one 3-D vector onto another using the axis–angle representation

### Optional Numba JIT acceleration
All numerically intensive functions are decorated with `cond_jit` from the bundled **pyConDec** package. When Numba is installed, functions are compiled to native machine code on first call and cached for subsequent calls; when Numba is absent, the identical Python/NumPy implementation is used transparently.

---

## Quick example

```python
import numpy as np
import so3          # or:  import so3tools  (same package, either name works)

# Euler vector → rotation matrix → back
omega = np.array([0.1, -0.2, 0.3])
R = so3.euler2rotmat(omega)
omega_recovered = so3.rotmat2euler(R)

# Quaternion round-trip
q = so3.mat2quat(R)
R2 = so3.quat2mat(q)

# Align two vectors
a = np.array([1.0, 0.0, 0.0])
b = np.array([0.0, 1.0, 0.0])
M = so3.rotmat_align_vector(a, b)   # M @ a  is parallel to  b
```

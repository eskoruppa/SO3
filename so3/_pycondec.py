"""
Shim that resolves pyConDec regardless of whether it is present as a
git-submodule (so3/pyConDec/) or as a pip-installed package.

Priority:
  1. Local submodule:  so3/pyConDec/pycondec  (recursive-clone / dev workflow)
  2. Installed package: pycondec               (pip-installed dependency)

Raises ImportError with a helpful message if neither is available.
"""

import sys
import pathlib
import warnings

_PYCONDEC_SUBMODULE = pathlib.Path(__file__).parent / "pyConDec"


def _ensure_pycondec() -> None:
    """Insert the submodule root into sys.path so that ``import pycondec``
    resolves to the local checkout. Does nothing if already importable or
    if the submodule is not present."""
    submodule_root = str(_PYCONDEC_SUBMODULE)
    if submodule_root not in sys.path:
        sys.path.insert(0, submodule_root)


try:
    # 1. Local submodule: so3/pyConDec/pycondec  (dev / recursive-clone workflow)
    if (_PYCONDEC_SUBMODULE / "pycondec").is_dir():
        _ensure_pycondec()
    from pycondec import cond_jit, cond_jitclass  # noqa: E402
except (ImportError, ModuleNotFoundError):
    try:
        # 2. Installed package
        from pycondec import cond_jit, cond_jitclass  # noqa: F811
    except (ImportError, ModuleNotFoundError) as e:
        raise ImportError(
            "pyConDec could not be found. Either:\n"
            "  • clone SO3 with --recurse-submodules so that so3/pyConDec/ is populated:\n"
            "      git clone --recurse-submodules https://github.com/eskoruppa/SO3.git\n"
            "  • or install pyConDec via pip:\n"
            "      pip install pyConDec\n"
            "    or from source:\n"
            "      git clone https://github.com/eskoruppa/pyConDec.git && cd pyConDec && pip install .\n"
        ) from e

def _numba_active() -> bool:
    """Return True if pyConDec resolved a usable numba.jit, i.e. cond_jit will
    actually JIT-compile. Mirrors pyConDec's own detection (its ``jit`` symbol
    is None when numba is missing or fails to import), so it stays accurate even
    when numba is installed but broken."""
    try:
        from pycondec.conditional_numba import jit  # noqa: E402
    except (ImportError, ModuleNotFoundError):
        return False
    return jit is not None

if not _numba_active():
    warnings.warn(
        "numba is not available; SO3's numba-accelerated routines are running "
        "in pure-Python fallback mode, which is substantially slower. Install "
        "the optional speedup with:  pip install 'SO3[numba]'",
        RuntimeWarning,
        stacklevel=2,
    )


__all__ = ["cond_jit", "cond_jitclass"]

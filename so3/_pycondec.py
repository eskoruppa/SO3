"""
Shim that resolves pyConDec regardless of whether it is present as a
git-submodule (so3/pyConDec/) or as a pip-installed package.

Priority:
  1. Local submodule:  so3/pyConDec/pycondec  (recursive-clone / dev workflow)
  2. Installed package: pycondec               (pip-installed dependency)

Raises ImportError with a helpful message if neither is available.
"""

try:
    from .pyConDec.pycondec import cond_jit, cond_jitclass
except (ImportError, ModuleNotFoundError):
    try:
        from pycondec import cond_jit, cond_jitclass
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

__all__ = ["cond_jit", "cond_jitclass"]

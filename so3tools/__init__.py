"""``so3tools`` is an alias for the :mod:`so3` import package.

The project is distributed on PyPI as ``so3tools``, but its canonical import
name is ``so3``. This thin shim lets either name be used interchangeably::

    import so3                     # canonical
    import so3tools                # alias — the same package object
    import so3tools.quaternions    # submodules resolve too

Both names resolve to the *same* module objects, so there is no duplicated
state and no import-time side effect (e.g. the missing-numba warning) runs
twice.
"""
import sys as _sys

import so3 as _so3

# Replace this module with the already-loaded ``so3`` package, and mirror every
# ``so3`` / ``so3.<sub>`` entry under the ``so3tools`` name so that submodule
# imports (``import so3tools.quaternions``) find the identical module object
# instead of re-importing it.
_sys.modules[__name__] = _so3
for _name, _module in list(_sys.modules.items()):
    if _name == "so3" or _name.startswith("so3."):
        _sys.modules["so3tools" + _name[len("so3"):]] = _module

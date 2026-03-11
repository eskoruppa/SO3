#!/usr/bin/env python3
"""
One-shot rename script:
  - In each library file, rename batch dispatchers to FUNC_batch.
  - In test files, update imports and call sites.
  - __init__.py is NOT touched here; it is rewritten separately.
"""
import re, sys
from pathlib import Path

BASE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Helper: rename function definitions in library files
# ---------------------------------------------------------------------------
def rename_defs(path: Path, old_to_new: dict[str, str]) -> None:
    text = path.read_text()
    for old, new in old_to_new.items():
        # rename 'def OLD(' → 'def NEW('
        text = text.replace(f"def {old}(", f"def {new}(")
    path.write_text(text)
    print(f"  updated defs in {path.name}")


# ---------------------------------------------------------------------------
# Helper: rename call-sites in a file, but NEVER rename *_single, *_batch,
# *_sv, *_flat variants.  Uses \b…\b(?!_) regex.
# ---------------------------------------------------------------------------
def rename_calls(path: Path, old_to_new: dict[str, str]) -> None:
    text = path.read_text()
    for old, new in old_to_new.items():
        # word-boundary match; NOT followed by _ (to protect _single etc.)
        pattern = r'\b' + re.escape(old) + r'\b(?!_)'
        text = re.sub(pattern, new, text)
    path.write_text(text)
    print(f"  updated calls  in {path.name}")


# ============================================================
# 1.  Cayley.py  –  4 batch dispatchers
# ============================================================
CAYLEY_RENAMES = {
    "cayley2rotmat":    "cayley2rotmat_batch",
    "rotmat2cayley":    "rotmat2cayley_batch",
    "se3_cayley2rotmat":"se3_cayley2rotmat_batch",
    "se3_rotmat2cayley":"se3_rotmat2cayley_batch",
}
rename_defs(BASE / "so3/Cayley.py", CAYLEY_RENAMES)

# ============================================================
# 2.  Euler.py  –  12 batch dispatchers
# ============================================================
EULER_RENAMES = {
    "euler2rotmat":           "euler2rotmat_batch",
    "rotmat2euler":           "rotmat2euler_batch",
    "sqrt_rot":               "sqrt_rot_batch",
    "midstep":                "midstep_batch",
    "right_jacobian":         "right_jacobian_batch",
    "left_jacobian":          "left_jacobian_batch",
    "inverse_right_jacobian": "inverse_right_jacobian_batch",
    "inverse_left_jacobian":  "inverse_left_jacobian_batch",
    "se3_euler2rotmat":       "se3_euler2rotmat_batch",
    "se3_rotmat2euler":       "se3_rotmat2euler_batch",
    "se3_eulers2rotmats":     "se3_eulers2rotmats_batch",
    "se3_rotmats2eulers":     "se3_rotmats2eulers_batch",
}
rename_defs(BASE / "so3/Euler.py", EULER_RENAMES)
# Fix the two wrapper bodies that call batch peer functions by name
euler_text = (BASE / "so3/Euler.py").read_text()
euler_text = euler_text.replace(
    "return se3_euler2rotmat_batch(X",
    "return se3_euler2rotmat_batch(X"   # already renamed by rename_defs above
)
(BASE / "so3/Euler.py").write_text(euler_text)

# ============================================================
# 3.  conversions.py  –  8 batch dispatchers
# ============================================================
CONV_RENAMES = {
    "cayley2euler":               "cayley2euler_batch",
    "euler2cayley":               "euler2cayley_batch",
    "cayley2euler_factor":        "cayley2euler_factor_batch",
    "euler2cayley_factor":        "euler2cayley_factor_batch",
    "cayley2euler_linearexpansion": "cayley2euler_linearexpansion_batch",
    "euler2cayley_linearexpansion": "euler2cayley_linearexpansion_batch",
    "splittransform_group2algebra": "splittransform_group2algebra_batch",
    "splittransform_algebra2group": "splittransform_algebra2group_batch",
}
rename_defs(BASE / "so3/conversions.py", CONV_RENAMES)
# Remove @cond_jit from the two that had it (cayley2euler_batch, euler2cayley_batch)
conv_text = (BASE / "so3/conversions.py").read_text()
conv_text = conv_text.replace(
    "@cond_jit(nopython=True, cache=True)\ndef cayley2euler_batch(",
    "def cayley2euler_batch("
)
conv_text = conv_text.replace(
    "@cond_jit(nopython=True, cache=True)\ndef euler2cayley_batch(",
    "def euler2cayley_batch("
)
(BASE / "so3/conversions.py").write_text(conv_text)

# ============================================================
# 4.  SE3.py  –  13 batch dispatchers
# ============================================================
SE3_RENAMES = {
    "se3_inverse":                        "se3_inverse_batch",
    "se3_triads2rotmat":                  "se3_triads2rotmat_batch",
    "se3_triads2euler":                   "se3_triads2euler_batch",
    "se3_midstep2triad":                  "se3_midstep2triad_batch",
    "se3_triad2midstep":                  "se3_triad2midstep_batch",
    "se3_triadxrotmat_midsteptrans":      "se3_triadxrotmat_midsteptrans_batch",
    "se3_triads2rotmat_midsteptrans":     "se3_triads2rotmat_midsteptrans_batch",
    "se3_transformation_triad2midstep":   "se3_transformation_triad2midstep_batch",
    "se3_transformation_midstep2triad":   "se3_transformation_midstep2triad_batch",
    "se3_algebra2group_lintrans":         "se3_algebra2group_lintrans_batch",
    "se3_group2algebra_lintrans":         "se3_group2algebra_lintrans_batch",
    "se3_algebra2group_stiffmat":         "se3_algebra2group_stiffmat_batch",
    "se3_group2algebra_stiffmat":         "se3_group2algebra_stiffmat_batch",
}
rename_defs(BASE / "so3/SE3.py", SE3_RENAMES)

# ============================================================
# 5.  Test files  –  update imports and call sites
#     Strategy: use rename_calls() which avoids _single/_sv/_flat/_batch
# ============================================================

# test_euler.py
# The import block currently names the functions without _batch.
# We rename all call-sites in the body; then patch the import block separately.
TEST_EULER = BASE / "test_euler.py"
rename_calls(TEST_EULER, EULER_RENAMES)
# Patch the import block: add _batch names alongside originals
euler_test = TEST_EULER.read_text()
OLD_IMPORT_EULER = """\
from so3 import (
    euler2rotmat_batch, rotmat2euler_batch,
    sqrt_rot_batch, midstep_batch,
    right_jacobian_batch, left_jacobian_batch,
    inverse_right_jacobian_batch, inverse_left_jacobian_batch,
    se3_euler2rotmat_batch, se3_rotmat2euler_batch,"""
NEW_IMPORT_EULER = """\
from so3 import (
    # _sv aliases (single-input JIT-callable, usable from numba)
    euler2rotmat, rotmat2euler,
    sqrt_rot, midstep,
    right_jacobian, left_jacobian,
    inverse_right_jacobian, inverse_left_jacobian,
    se3_euler2rotmat, se3_rotmat2euler,
    # batch dispatchers (Python, accept any shape)
    euler2rotmat_batch, rotmat2euler_batch,
    sqrt_rot_batch, midstep_batch,
    right_jacobian_batch, left_jacobian_batch,
    inverse_right_jacobian_batch, inverse_left_jacobian_batch,
    se3_euler2rotmat_batch, se3_rotmat2euler_batch,"""
euler_test = euler_test.replace(OLD_IMPORT_EULER, NEW_IMPORT_EULER)
TEST_EULER.write_text(euler_test)

# test_cayley.py
TEST_CAYLEY = BASE / "test_cayley.py"
rename_calls(TEST_CAYLEY, CAYLEY_RENAMES)
cayley_test = TEST_CAYLEY.read_text()
OLD_IMPORT_CAYLEY = """\
from so3 import (
    cayley2rotmat_batch, rotmat2cayley_batch,
    se3_cayley2rotmat_batch, se3_rotmat2cayley_batch,"""
NEW_IMPORT_CAYLEY = """\
from so3 import (
    # _sv aliases (single-input JIT-callable)
    cayley2rotmat, rotmat2cayley,
    se3_cayley2rotmat, se3_rotmat2cayley,
    # batch dispatchers
    cayley2rotmat_batch, rotmat2cayley_batch,
    se3_cayley2rotmat_batch, se3_rotmat2cayley_batch,"""
cayley_test = cayley_test.replace(OLD_IMPORT_CAYLEY, NEW_IMPORT_CAYLEY)
TEST_CAYLEY.write_text(cayley_test)

# test_conversions.py
TEST_CONV = BASE / "test_conversions.py"
# conversions test imports directly from so3.conversions — rename those too
rename_calls(TEST_CONV, CONV_RENAMES)

# The import block uses the old names; patch it
conv_test = TEST_CONV.read_text()
OLD_IMPORT_CONV = """\
from so3.conversions import (
    # batch
    cayley2euler_batch,
    euler2cayley_batch,
    cayley2euler_factor_batch,
    euler2cayley_factor_batch,
    cayley2euler_linearexpansion_batch,
    euler2cayley_linearexpansion_batch,
    splittransform_group2algebra_batch,
    splittransform_algebra2group_batch,"""
NEW_IMPORT_CONV = """\
from so3.conversions import (
    # _sv aliases (single-input JIT-callable)
    cayley2euler_single as cayley2euler,
    euler2cayley_single as euler2cayley,
    cayley2euler_factor_single as cayley2euler_factor,
    euler2cayley_factor_single as euler2cayley_factor,
    cayley2euler_linearexpansion_single as cayley2euler_linearexpansion,
    euler2cayley_linearexpansion_single as euler2cayley_linearexpansion,
    splittransform_group2algebra_single as splittransform_group2algebra,
    splittransform_algebra2group_single as splittransform_algebra2group,
    # batch dispatchers
    cayley2euler_batch,
    euler2cayley_batch,
    cayley2euler_factor_batch,
    euler2cayley_factor_batch,
    cayley2euler_linearexpansion_batch,
    euler2cayley_linearexpansion_batch,
    splittransform_group2algebra_batch,
    splittransform_algebra2group_batch,"""
conv_test = conv_test.replace(OLD_IMPORT_CONV, NEW_IMPORT_CONV)
TEST_CONV.write_text(conv_test)

# test_se3.py  –  accesses via so3.FUNC  (module-attribute style)
# We need to change so3.FUNC → so3.FUNC_batch for all batch dispatchers.
TEST_SE3 = BASE / "test_se3.py"
se3_test = TEST_SE3.read_text()
# Build mapping for so3.FUNC → so3.FUNC_batch
# Note: so3.euler2rotmat is used for SINGLE inputs in helpers → keep as-is
# (it maps to _sv which handles single).  But se3_* dispatchers need _batch.
SE3_MODULE_RENAMES = {f"so3.{old}": f"so3.{new}" for old, new in SE3_RENAMES.items()}
for old_ref, new_ref in SE3_MODULE_RENAMES.items():
    se3_test = se3_test.replace(old_ref, new_ref)
TEST_SE3.write_text(se3_test)

print("\nAll renames complete.")

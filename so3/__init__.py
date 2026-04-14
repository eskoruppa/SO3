"""
SO3
=====

A package for various rotation operations

"""

##########################################################################################################
# Cayley imports
##########################################################################################################

from .Cayley import (
    cayley2rotmat_batch, rotmat2cayley_batch, se3_cayley2rotmat_batch,  se3_rotmat2cayley_batch,
    # single-input variants (original implementations)
    cayley2rotmat_single, rotmat2cayley_single,
    se3_cayley2rotmat_single, se3_rotmat2cayley_single,
)
from .Cayley import _cayley2rotmat_sv as cayley2rotmat
from .Cayley import _rotmat2cayley_sv as rotmat2cayley
from .Cayley import _se3_cayley2rotmat_sv as se3_cayley2rotmat
from .Cayley import _se3_rotmat2cayley_sv as se3_rotmat2cayley

##########################################################################################################
# Conversion imports
##########################################################################################################

from .conversions import (
    cayley2euler_batch,
    cayley2euler_factor_batch,
    cayley2euler_linearexpansion_batch,
    euler2cayley_batch,
    euler2cayley_factor_batch,
    euler2cayley_linearexpansion_batch,
    splittransform_algebra2group_batch,
    splittransform_group2algebra_batch,
)

from .conversions import _cayley2euler_sv as cayley2euler
from .conversions import _cayley2euler_factor_sv as cayley2euler_factor
from .conversions import _cayley2euler_linearexpansion_sv as cayley2euler_linearexpansion
from .conversions import _euler2cayley_sv as euler2cayley
from .conversions import _euler2cayley_factor_sv as euler2cayley_factor
from .conversions import _euler2cayley_linearexpansion_sv as euler2cayley_linearexpansion
from .conversions import _splittransform_algebra2group_sv as splittransform_algebra2group
from .conversions import _splittransform_group2algebra_sv as splittransform_group2algebra

from .conversions import (
    # single-vector variants (JIT-callable)
    cayley2euler_single,
    cayley2euler_factor_single,
    cayley2euler_linearexpansion_single,
    euler2cayley_single,
    euler2cayley_factor_single,
    euler2cayley_linearexpansion_single,
    splittransform_algebra2group_single,
    splittransform_group2algebra_single,
)

##########################################################################################################
# Euler imports
##########################################################################################################

from .Euler import (
    euler2rotmat_batch,
    midstep_batch,
    rotmat2euler_batch,
    se3_euler2rotmat_batch,
    se3_rotmat2euler_batch,
    se3_eulers2rotmats,
    se3_rotmats2eulers,
    sqrt_rot_batch,
    right_jacobian_batch,
    left_jacobian_batch,
    inverse_right_jacobian_batch,
    inverse_left_jacobian_batch,
)

from .Euler import _euler2rotmat_sv as euler2rotmat
from .Euler import _midstep_sv as midstep
from .Euler import _rotmat2euler_sv as rotmat2euler
from .Euler import _se3_euler2rotmat_sv as se3_euler2rotmat
from .Euler import _se3_rotmat2euler_sv as se3_rotmat2euler
from .Euler import _sqrt_rot_sv as sqrt_rot
from .Euler import _right_jacobian_sv as right_jacobian
from .Euler import _left_jacobian_sv as left_jacobian
from .Euler import _inverse_right_jacobian_sv as inverse_right_jacobian
from .Euler import _inverse_left_jacobian_sv as inverse_left_jacobian


from .Euler import (
    # single-vector variants (JIT-callable)
    euler2rotmat_single,
    rotmat2euler_single,
    sqrt_rot_single,
    midstep_single,
    right_jacobian_single,
    left_jacobian_single,
    inverse_right_jacobian_single,
    inverse_left_jacobian_single,
    se3_euler2rotmat_single,
    se3_rotmat2euler_single,
)

from .generators import (
    generator1, generator2, generator3, 
    hat_map_batch, vec_map_batch,
    # single-input variants (original implementations)
    # JIT-callable _sv helpers (for use inside other @cond_jit functions)
    _hat_map_sv, _vec_map_sv,
)

from .generators import _hat_map_sv as hat_map
from .generators import _vec_map_sv as vec_map

from .generators import (
    hat_map_single, vec_map_single,
)

from .extend_euler import extend_euler
from .matrices import dots
from ._pycondec import cond_jit


from .SE3 import (
    se3_algebra2group_lintrans_batch,
    se3_algebra2group_stiffmat_batch,
    se3_group2algebra_lintrans_batch,
    se3_group2algebra_stiffmat_batch,
    se3_inverse_batch,
    se3_midstep2triad_batch,
    se3_transformation_midstep2triad_batch,
    se3_transformation_triad2midstep_batch,
    se3_triad2midstep_batch,
    se3_triads2euler_batch,
    se3_triads2rotmat_batch,
    se3_triads2rotmat_midsteptrans_batch,
    se3_triadxrotmat_midsteptrans_batch,
)

from .SE3 import _se3_algebra2group_lintrans_sv as se3_algebra2group_lintrans
from .SE3 import _se3_algebra2group_stiffmat_sv as se3_algebra2group_stiffmat
from .SE3 import _se3_group2algebra_lintrans_sv as se3_group2algebra_lintrans
from .SE3 import _se3_group2algebra_stiffmat_sv as se3_group2algebra_stiffmat
from .SE3 import _se3_inverse_sv as se3_inverse
from .SE3 import _se3_midstep2triad_sv as se3_midstep2triad
from .SE3 import _se3_transformation_midstep2triad_sv as se3_transformation_midstep2triad
from .SE3 import _se3_transformation_triad2midstep_sv as se3_transformation_triad2midstep
from .SE3 import _se3_triad2midstep_sv as se3_triad2midstep
from .SE3 import _se3_triads2euler_sv as se3_triads2euler
from .SE3 import _se3_triads2rotmat_sv as se3_triads2rotmat
from .SE3 import _se3_triads2rotmat_midsteptrans_sv as se3_triads2rotmat_midsteptrans
from .SE3 import _se3_triadxrotmat_midsteptrans_sv as se3_triadxrotmat_midsteptrans

from .SE3 import (
    # _single originals
    se3_inverse_single,
    se3_triads2rotmat_single,
    se3_triads2euler_single,
    se3_midstep2triad_single,
    se3_triad2midstep_single,
    se3_triadxrotmat_midsteptrans_single,
    se3_triads2rotmat_midsteptrans_single,
    se3_transformation_triad2midstep_single,
    se3_transformation_midstep2triad_single,
    se3_algebra2group_lintrans_single,
    se3_group2algebra_lintrans_single,
    se3_algebra2group_stiffmat_single,
    se3_group2algebra_stiffmat_single,
)

from .se3_junction_methods import (
    X_inv,
    X2g,
    g2X,
    X2glh,
    X2grh,
    glh2X,
    grh2X,
    g2glh,
    g2grh,
    glh2g,
    grh2g,
    g2glh_inv,
    g2grh_inv,
    glh2g_inv,
    grh2g_inv,
    X2g_inv,
    X2glh_inv,
    X2grh_inv,
    A_rev,
    A_lh,
    A_rh,
)

from .quaternions import (
    quat2mat_numba,
    quat2mat,
    mat2quat_numba,
    mat2quat,
    quats2mats,
    mats2quats,
    quats2rotmats #depricated!
)

from .rotation_methods import rotmat_align_vector


from .transforms.transform_algebra2group import (
    algebra2group_lintrans,
    algebra2group_params,
    algebra2group_stiffmat,
    group2algebra_lintrans,
    group2algebra_stiffmat,
    group2algebra_params,
)

from .transforms.transform_cayley2euler import (
    se3_euler2cayley,
    se3_cayley2euler,
    se3_cayley2euler_lintrans,
    se3_euler2cayley_lintrans,
    se3_cayley2euler_stiffmat,
    se3_euler2cayley_stiffmat,
)

from .transforms.transform_midstep2triad import (
    midstep2triad,
    triad2midstep,
    midstep2triad_lintrans,
    triad2midstep_lintrans,
    midstep2triad_stiffmat,
    triad2midstep_stiffmat,
)

from .transforms.transform_marginals import (
    matrix_marginal,
    vector_marginal,
    matrix_marginal_assignment,
    vector_marginal_assignment,
    unwrap_wildtypes,
    matrix_blockmarginal,
    vector_blockmarginal,
    matrix_rotmarginal,
    matrix_transmarginal,
    vector_rotmarginal,
    vector_transmarginal,
    marginal_schur_complement,
)

from .transforms.transform_statevec import (
    statevec2vecs,
    vecs2statevec,
)

from .transforms.transform_units import array_conversion


# legacy methods
from .SO3Methods import phi2rotx, phi2roty, phi2rotz

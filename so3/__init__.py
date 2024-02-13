"""
SO3
=====

A package for various rotation operations

"""

from .pyConDec.pycondec import cond_jit
from .generators import hat_map, vec_map
from .generators import generator1, generator2, generator3
from .Cayley import cayley2rotmat, rotmat2cayley, se3_cayley2rotmat, se3_rotmat2cayley
from .Euler import euler2rotmat, rotmat2euler, se3_euler2rotmat, se3_rotmat2euler
from .conversions import cayley2euler, cayley2euler_factor
from .conversions import euler2cayley, euler2cayley_factor
from .conversions import cayley2euler_linearexpansion, euler2cayley_linearexpansion
from .conversions import splittransform_group2algebra, splittransform_algebra2group
from .matrices import dots
from .Euler import sqrt_rot, midstep
from .SE3 import se3_inverse, se3_triads2rotmat, se3_triads2euler
from .SE3 import se3_triadxrotmat_midsteptrans, se3_triads2rotmat_midsteptrans, se3_transformation_triad2midstep, se3_transformation_midstep2triad
from .SE3 import se3_triad2midstep, se3_midstep2triad
from .SE3 import se3_algebra2group_lintrans, se3_group2algebra_lintrans
from .SE3 import se3_algebra2group_stiffmat, se3_group2algebra_stiffmat


# legacy method
from .SO3Methods  import  phi2rotx, phi2roty, phi2rotz


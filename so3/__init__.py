"""
SO3
=====

A package for various rotation operationss

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
from .SE3 import se3_inverse, se3_triads2rotmat, sqrt_rot
from .SE3 import se3_triadxrotmat_midsteptrans, se3_triads2rotmat_midsteptrans, se3_triad_normal2midsteptrans, se3_triad_midsteptrans2normal

# legacy method
from .SO3Methods  import  phi2rotx, phi2roty, phi2rotz


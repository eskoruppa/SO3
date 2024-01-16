"""
pyConDec
=====

Module for conditional decorators in python

"""

from .pyConDec.pycondec import cond_jit
from .generators import hat_map, vec_map
from .generators import generator1, generator2, generator3
from .Cayley import cayley2rotmat, rotmat2cayley, cayley2rotmat_se3, rotmat2cayley_se3
from .Euler import euler2rotmat, rotmat2euler, euler2rotmat_se3, rotmat2euler_se3
from .conversions import cayley2euler, cayley2euler_factor
from .conversions import euler2cayley, euler2cayley_factor
from .conversions import cayley2euler_linearexpansion, euler2cayley_linearexpansion
from .conversions import splittransform_group2algebra, splittransform_algebra2group
from .matrices import dots

# legacy method
from .SO3Methods  import  phi2rotx, phi2roty, phi2rotz


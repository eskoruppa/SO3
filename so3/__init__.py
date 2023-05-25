"""
pyConDec
=====

Module for conditional decorators in python

"""

from .pyConDec.pycondec import cond_jit
from .generators import hat_map, vec_map, generator1, generator2, generator3
from .Cayley import cayley2rotmat, rotmat2cayley
from .Euler import euler2rotmat, rotmat2euler
from .conversions import cayley2euler, cayley2euler_factor, euler2cayley, euler2cayley_factor

# legacy method
from .SO3Methods  import  phi2rotx, phi2roty, phi2rotz


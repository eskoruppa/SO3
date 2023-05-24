"""
pyConDec
=====

Module for conditional decorators in python

"""

from .pyConDec.pycondec import cond_jit
from .generators import hat_map, vec_map, generator1, generator2, generator3
from .Cayley import cayley2rotmat, rotmat2cayley
from .Cayley import rotx,roty,rotz
from .Euler import euler2rotmat, rotmat2euler

# legacy method
from .SO3Methods  import  phi2rotz


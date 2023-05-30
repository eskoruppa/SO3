"""
pyConDec
=====

Module for conditional decorators in python

"""

from .pyConDec.pycondec import cond_jit
from .generators import hat_map, vec_map, generator1, generator2, generator3
from .Cayley import cayley2rotmat, rotmat2cayley
from .Euler import euler2rotmat, rotmat2euler
from .conversions import cayley2euler, cayley2euler_factor
from .conversions import euler2cayley, euler2cayley_factor
from .conversions import cayley2euler_linearexpansion, euler2cayley_linearexpansion
from .conversions import splittransform_group2algebra, splittransform_algebra2group

from .transforms import statevec2vecs, vecs2statevec, eulers2rotmats, rotmats2eulers
from .transforms import eulers2rotmats_SO3fluct
from .transforms import cayleys2rotmats, rotmats2cayleys, vecs2rotmats, rotmats2vecs
from .transforms import rotmats2triads, triads2rotmats
from .transforms import triads2positions
from .transforms import eulers2cayleys, cayleys2eulers, cayleys2eulers_lintrans, eulers2cayleys_lintrans
from .transforms import splittransform_group2algebra, splittransform_algebra2group


from .matrices import dots

# legacy method
from .SO3Methods  import  phi2rotx, phi2roty, phi2rotz


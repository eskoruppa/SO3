#!/bin/env python3

import numpy as np
from .generators import hat_map
from .pyConDec.pycondec import cond_jit

from .conversions import euler2rotmat, rotmat2euler, cayley2rotmat, rotmat2cayley
from .conversions import euler2cayley, cayley2euler, cayley2euler_linearexpansion, euler2cayley_linearexpansion
from .conversions import splittransform_group2algebra, splittransform_algebra2group

##########################################################################################################
##########################################################################################################
############### Conversions between Lie Algebra and Lie Group coordinates ################################
##########################################################################################################
##########################################################################################################

##########################################################################################################
############### Conversions between state vectors (3N) and 3-vectors (N,3) ###############################
##########################################################################################################

def statevec2vecs(statevec: np.ndarray) -> np.ndarray:
    """ reshapes configuration of full state (3N) vectors into 3-vectors. Turns last dimensions from (3N) to (N,3)

    Args:
        vecs (np.ndarray): collection of state vectors

    Returns:
        np.ndarray: collection of 3-vectors
    """
    if statevec.shape[-1] == 3:
        return statevec
    if len(statevec.shape) == 1:
        if len(statevec)%3 != 0:
            raise ValueError(f'statevec is inconsistent with list of euler vectors. THe number of entries needs to be a multiple of 3. len(statevec)%3 = {len(statevec)%3}')
        return np.reshape(statevec,(len(statevec)//3,3))
    if statevec.shape[-1]%3 != 0:
        raise ValueError(f'statevec is inconsistent with list of euler vectors. THe number of entries needs to be a multiple of 3. statevec.shape[-1]%3 = {statevec.shape[-1]%3}')
    return np.reshape(statevec,(statevec.shape[:-1] + (statevec.shape[-1]//3,3)))

def vecs2statevec(vecs: np.ndarray) -> np.ndarray:
    """ reshapes configuration of 3-vectors into full state (3N) vectors. Turns last dimensions from (N,3) to (3N,)

    Args:
        vecs (np.ndarray): collection of 3-vectors

    Returns:
        np.ndarray: collection of state vectors 
    """
    if vecs.shape[-1] != 3:
        raise ValueError(f'Expected collection of 3-vectors. Vectors of dimension {vecs.shape[-1]} given.')
    newshape = list(vecs.shape)
    newshape = newshape[:-1]
    newshape[-1] *= 3
    return np.reshape(vecs,tuple(newshape))


##########################################################################################################
############### Conversion between Euler vectors and rotation matrices ###################################
##########################################################################################################

def eulers2rotmats(eulers: np.ndarray) -> np.ndarray:
    """Converts configuration of euler vectors into collection of rotation matrices

    Args:
        eulers (np.ndarray): Collection of euler vectors (...,N,3)

    Returns:
        np.ndarray: collection of rotation matrices (...,N,3,3)
    """
    rotmats = np.zeros(eulers.shape + (3,))
    if len(eulers.shape) > 2:
        for i in range(len(eulers)):
            rotmats[i] = eulers2rotmats(eulers[i])
        return rotmats
    for i,euler in enumerate(eulers):
        rotmats[i] = euler2rotmat(euler)
    return rotmats


def rotmats2eulers(rotmats: np.ndarray) -> np.ndarray:
    """Converts collection of rotation matrices into collection of euler vectors

    Args:
        rotmats (np.ndarray): collection of rotation matrices (...,N,3,3)

    Returns:
        np.ndarray: Collection of euler vectrs (...,N,3)
    """
    eulers = np.zeros(rotmats.shape[:-1])
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            eulers[i] = rotmats2eulers(rotmats[i])
        return eulers
    for i,rotmat in enumerate(rotmats):
        eulers[i] = rotmat2euler(rotmat)
    return eulers


def eulers2rotmats_SO3fluct(eulers_gs: np.ndarray,eulers_fluct: np.ndarray) -> np.ndarray:
    """Converts configuration of euler vectors into collection of rotation matrices

    Args:
        eulers_gs (np.ndarray): Groundstate euler vectors (N,3)
        eulers_fluct (np.ndarray): Collection of fluctuating component euler vectors (...,N,3)

    Returns:
        np.ndarray: collection of rotation matrices (...,N,3,3)
    """
    
    def _eulers2rotmats_SO3fluct(eulers_gs_rotmat: np.ndarray,eulers_fluct: np.ndarray) -> np.ndarray:
        rotmats = np.zeros(eulers_fluct.shape + (3,))
        if len(eulers_fluct.shape) > 2:
            for i in range(len(eulers_fluct)):
                rotmats[i] = _eulers2rotmats_SO3fluct(eulers_gs_rotmat,eulers_fluct[i])
            return rotmats
        
        for i,euler in enumerate(eulers_fluct):
            rotmats[i] = np.matmul(eulers_gs_rotmat[i],euler2rotmat(euler))
        return rotmats

    eulers_gs_rotmat = eulers2rotmats(eulers_gs)
    return _eulers2rotmats_SO3fluct(eulers_gs_rotmat,eulers_fluct)



##########################################################################################################
############### Conversion between Cayley vectors and rotation matrices ###################################
##########################################################################################################

def cayleys2rotmats(cayleys: np.ndarray) -> np.ndarray:
    """Converts configuration of euler vectors into collection of rotation matrices

    Args:
        eulers (np.ndarray): Collection of euler vectors (...,N,3)

    Returns:
        np.ndarray: collection of rotation matrices (...,N,3,3)
    """
    rotmats = np.zeros(cayleys.shape + (3,))
    if len(cayleys.shape) > 2:
        for i in range(len(cayleys)):
            rotmats[i] = cayleys2rotmats(cayleys[i])
        return rotmats
    for i,cayley in enumerate(cayleys):
        rotmats[i] = cayley2rotmat(cayley)
    return rotmats


def rotmats2cayleys(rotmats: np.ndarray) -> np.ndarray:
    """Converts collection of rotation matrices into collection of euler vectors

    Args:
        rotmats (np.ndarray): collection of rotation matrices (...,N,3,3)

    Returns:
        np.ndarray: Collection of euler vectors (...,N,3) 
    """
    cayleys = np.zeros(rotmats.shape[:-1])
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            cayleys[i] = rotmats2cayleys(rotmats[i])
        return cayleys
    for i,rotmat in enumerate(rotmats):
        cayleys[i] = rotmat2cayley(rotmat)
    return cayleys

##########################################################################################################
############### Conversion between vectors and rotation matrices #########################################
##########################################################################################################

def vecs2rotmats(vecs: np.ndarray,
                 rotation_map='euler') -> np.ndarray:
    """Converts configuration of vectors into collection of rotation matrices

    Args:
        vecs (np.ndarray): Collection of rotational vectors (...,N,3)
        rotation_map (str): selected map between rotation rotation coordinates and rotation matrix. 
                Options:    - cayley: default cnDNA map (Cayley map)
                            - euler:  Axis angle representation.  

    Returns:
        np.ndarray: collection of rotation matrices (...,N,3,3)
    """
    if rotation_map == 'euler':
        return eulers2rotmats(vecs)
    elif rotation_map == 'cayley':
        return cayleys2rotmats(vecs)
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')

def rotmats2vecs(rotmats: np.ndarray,
                    rotation_map='euler') -> np.ndarray:
    """Converts collection of rotation matrices into collection of vectors

    Args:
        rotmats (np.ndarray): collection of rotation matrices (...,N,3,3)
        rotation_map (str): selected map between rotation rotation coordinates and rotation matrix. 
                Options:    - cayley: default cnDNA map (Cayley map)
                            - euler:  Axis angle representation.  

    Returns:
        np.ndarray: Collection of vectors (...,N,3) 
    """
    if rotation_map == 'euler':
        return rotmats2eulers(rotmats)
    elif rotation_map == 'cayley':
        return rotmats2cayleys(rotmats)
    else:
        raise ValueError(f'Unknown rotation_map "{rotation_map}"')

##########################################################################################################
############### Conversion between rotation matrices and triads ##########################################
##########################################################################################################

def rotmats2triads(rotmats: np.ndarray, first_triad=None) -> np.ndarray:
    """Converts collection of rotation matrices into collection of triads

    Args:
        rotmats (np.ndarray): set of rotation matrices that constitute the local junctions in the chain of triads. (...,N,3,3)
        first_triad (None or np.ndarray): rotation of first triad. Should be none or single triad. For now only supports identical rotation for all snapshots.

    Returns:
        np.ndarray: set of triads (...,N+1,3,3)
    """
    sh = list(rotmats.shape)
    sh[-3] += 1
    triads = np.zeros(tuple(sh))
    if len(rotmats.shape) > 3:
        for i in range(len(rotmats)):
            triads[i] = rotmats2triads(rotmats[i])
        return triads
    
    if first_triad is None:
        first_triad = np.eye(3)
    assert first_triad.shape == (3,3), f'invalid shape of triad {first_triad.shape}. Triad shape needs to be (3,3).'

    triads[0] = first_triad
    for i,rotmat in enumerate(rotmats):
        triads[i+1] = np.matmul(triads[i],rotmat)
    return triads


def triads2rotmats(triads: np.ndarray) -> np.ndarray:
    """Converts set of triads into set of rotation matrices

    Args:
        triads (np.ndarray): set of triads (...,N+1,3,3)

    Returns:
        np.ndarray: set of rotation matrices (...,N,3,3)
    """
    sh = list(triads.shape)
    sh[-3] -= 1
    rotmats = np.zeros(tuple(sh))
    if len(triads.shape) > 3:
        for i in range(len(triads)):
            rotmats[i] = triads2rotmats(triads[i])
        return rotmats
    
    for i in range(len(triads)-1):
        rotmats[i] = np.matmul(triads[i].T,triads[i+1])
    return rotmats

##########################################################################################################
############### Generate positions from triads ###########################################################
##########################################################################################################

def triads2positions(triads: np.ndarray, disc_len=0.34) -> np.ndarray:
    """generates a set of position vectors from a set of triads

    Args:
        triads (np.ndarray): set of trads (...,N,3,3)
        disc_len (float): discretization length

    Returns:
        np.ndarray: set of position vectors (...,N,3)
    """
    pos = np.zeros(triads.shape[:-1])
    if len(triads.shape) > 3:
        for i in range(len(triads)):
            pos[i] = triads2positions(triads[i])
        return pos
    pos[0] = np.zeros(3)
    for i in range(len(triads)-1):
        pos[i+1] = pos[i] + triads[i,:,2] * disc_len
    return pos

##########################################################################################################
##########################################################################################################
############### Conversion between Euler and Cayley (Rodrigues) coordinates ##############################  
##########################################################################################################
##########################################################################################################

def eulers2cayleys(eulers: np.ndarray) -> np.ndarray:
    """Converts set of Euler vectors (axis angle rotation vectors) into Rodrigues 
    vectors (Cayley vectors)

    Args:
        eulers (np.ndarray): set of Euler vectors

    Returns:
        np.ndarray: set of Rodrigues vectors
    """
    assert eulers.shape[-1] == 3, f'Expected set of 3-vectors. Instead received shape {eulers.shape}'

    cayleys = np.zeros(eulers.shape)
    if (len(eulers.shape) > 2):
        for i in range(len(eulers)):
            cayleys[i] = eulers2cayleys(eulers[i])
        return cayleys

    for i,euler in enumerate(eulers):
        cayleys[i] = euler2cayley(euler)
    return cayleys

def cayleys2eulers(cayleys: np.ndarray) -> np.ndarray:
    """Converts set of rodrigues vectors (Cayley vectors) into Euler vectors (axis angle 
    rotation vectors)

    Args:
        cayleys (np.ndarray): set of Rodreigues vectors

    Returns:
        np.ndarray: set of Euler vectors
    """
    assert cayleys.shape[-1] == 3, f'Expected set of 3-vectors. Instead received shape {cayleys.shape}'

    eulers = np.zeros(cayleys.shape)
    if (len(cayleys.shape) > 2):
        for i in range(len(cayleys)):
            eulers[i] = cayleys2eulers(cayleys[i])
        return eulers
    for i,cayley in enumerate(cayleys):
        eulers[i] = cayley2euler(cayley)
    return eulers


def cayleys2eulers_lintrans(cayleys: np.ndarray) -> np.ndarray:
    """Linearization of the transformation from Cayley to Euler vector around a given 
    groundstate vector

    Args:
        cayley (np.ndarray): Set of Cayley vectors around which the transformation is linearly expanded

    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    if len(cayleys.shape) > 1:
        if len(cayleys.shape) > 2:
            raise ValueError(f'Expected array of dimension 1 or 3. Given array of shape {cayleys.shape}')
        if cayleys.shape[0] > 1:
            raise ValueError(f'Two-dimensional arrays are expected to contain a single entry in the first dimension. Given array of shape {cayleys.shape}.')
        cayleys = cayleys[0]

    trans = np.zeros((len(cayleys),)*2)
    vecs = statevec2vecs(cayleys)
    for i,vec in enumerate(vecs):
        trans[3*i:3*(i+1),3*i:3*(i+1)] = cayley2euler_linearexpansion(vec)
    return trans


def eulers2cayleys_lintrans(eulers: np.ndarray) -> np.ndarray:
    """Linearization of the transformation from Euler to Cayley vector around a 
    given groundstate vector

    Args:
        cayley (np.ndarray): Set of Euler vectors around which the transformation is linearly expanded

    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    if len(eulers.shape) > 1:
        if len(eulers.shape) > 2:
            raise ValueError(f'Expected array of dimension 1 or 3. Given array of shape {eulers.shape}')
        if eulers.shape[0] > 1:
            raise ValueError(f'Two-dimensional arrays are expected to contain a single entry in the first dimension. Given array of shape {eulers.shape}.')
        eulers = eulers[0]

    trans = np.zeros((len(eulers),)*2)
    vecs = statevec2vecs(eulers)
    for i,vec in enumerate(vecs):
        trans[3*i:3*(i+1),3*i:3*(i+1)] = euler2cayley_linearexpansion(vec)
    return trans

##########################################################################################################
##########################################################################################################
############### Change Splitting between static and dynamic components ###################################
##########################################################################################################
##########################################################################################################

def splittransform_group2algebra(Theta_groundstate: np.ndarray) -> np.ndarray:
    """
    Linear transformation that maps dynamic component in group splitting representation 
    (R = D*S = exp(hat(Theta_0))exp(hat(Delta))), with D,S \in SO(3) to lie algebra splitting 
    representation R = exp(hat(Theta_0) + hat(Delta')). Linear transformation T transforms Delta
    into Delta' as T*Delta = Delta'.

    Args:
        Theta_0 (np.ndarray): static rotational component expressed in Axis angle parametrization (Euler vector)
        Has to be expressed in radians. (3N array expected)

    Returns:
        float: Linear transformation matrix T (3Nx3N) that transforms Delta into Delta': T*Delta = Delta'
    """
    N = len(Theta_groundstate)
    T = np.zeros((N,)*2)
    for i in range(N//3):
        T0 = Theta_groundstate[i*3:(i+1)*3]
        T[i*3:(i+1)*3,i*3:(i+1)*3] = splittransform_group2algebra(T0)
    return T

def splittransform_algebra2group(Theta_groundstate: np.ndarray) -> np.ndarray:
    """
    Linear transformation that maps dynamic component in lie algebra splitting representation R = exp(hat(Theta_0) + hat(Delta')) to group splitting representation 
    (R = D*S = exp(hat(Theta_0))exp(hat(Delta))), with D,S \in SO(3) t. Linear transformation T transforms Delta
    into Delta' as T'*Delta' = Delta. Currently this is defined as the inverse of the transformation
    defined in the method splittransform_group2algebra

    Args:
        Theta_0 (np.ndarray): static rotational component expressed in Axis angle parametrization (Euler vector)
        Has to be expressed in radians. (3N array expected)

    Returns:
        float: Linear transformation matrix T' (3Nx3N) that transforms Delta into Delta': T'*Delta' = Delta
    """
    N = len(Theta_groundstate)
    T = np.zeros((N,)*2)
    for i in range(N//3):
        T0 = Theta_groundstate[i*3:(i+1)*3]
        T[i*3:(i+1)*3,i*3:(i+1)*3] = splittransform_algebra2group(T0)
    return T
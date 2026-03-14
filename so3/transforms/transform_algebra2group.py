from __future__ import annotations

import numpy as np

from .._pycondec import cond_jit
from ..conversions import splittransform_algebra2group_batch, splittransform_group2algebra_batch
from ..conversions import _splittransform_algebra2group_sv, _splittransform_group2algebra_sv
from ..Euler import euler2rotmat_batch
from ..Euler import _euler2rotmat_sv
from ..generators import hat_map_batch
from ..generators import _hat_map_sv
from .transform_midstep2triad import midstep2triad, triad2midstep

    
##########################################################################################################
########## JIT kernels (private) – operate on already-validated, always-2D inputs #######################
##########################################################################################################

@cond_jit(nopython=True, cache=True)
def _algebra2group_lintrans_jit(
    groundstate_algebra: np.ndarray,   # always 2D, (N, 3) or (N, 6)
    translations_included: bool,
    rotation_first: bool,
    translation_as_midstep: np.ndarray,  # np.bool_ array of length N
) -> np.ndarray:
    N = groundstate_algebra.shape[0]
    dim = N * 6 if translations_included else N * 3
    HX = np.eye(dim)
    if not translations_included:
        for i in range(N):
            HX[3*i:3*(i+1), 3*i:3*(i+1)] = _splittransform_algebra2group_sv(groundstate_algebra[i])
    else:
        rot_start   = 0 if rotation_first else 3
        trans_start = 3 if rotation_first else 0
        for i in range(N):
            vec     = groundstate_algebra[i]
            Omega_0 = vec[rot_start:rot_start+3]
            zeta_0  = vec[trans_start:trans_start+3]
            H   = _splittransform_algebra2group_sv(Omega_0)
            sid = 6 * i
            rfr = sid + rot_start;  rto = rfr + 3
            tfr = sid + trans_start; tto = tfr + 3
            HX[rfr:rto, rfr:rto] = H
            if translation_as_midstep[i]:
                sqrtS_transp      = _euler2rotmat_sv(-0.5 * Omega_0)
                zeta_0_hat_transp = _hat_map_sv(-zeta_0)
                H_half            = _splittransform_algebra2group_sv(0.5 * Omega_0)
                HX[tfr:tto, rfr:rto] = 0.5 * sqrtS_transp @ zeta_0_hat_transp @ H_half
                HX[tfr:tto, tfr:tto] = sqrtS_transp
            else:
                HX[tfr:tto, tfr:tto] = _euler2rotmat_sv(-Omega_0)
    return HX


@cond_jit(nopython=True, cache=True)
def _group2algebra_lintrans_jit(
    groundstate_group: np.ndarray,   # always 2D, (N, 3) or (N, 6)
    translations_included: bool,
    rotation_first: bool,
    translation_as_midstep: np.ndarray,  # np.bool_ array of length N
) -> np.ndarray:
    N = groundstate_group.shape[0]
    dim = N * 6 if translations_included else N * 3
    HX_inv = np.eye(dim)
    if not translations_included:
        for i in range(N):
            HX_inv[3*i:3*(i+1), 3*i:3*(i+1)] = _splittransform_group2algebra_sv(groundstate_group[i])
    else:
        rot_start   = 0 if rotation_first else 3
        trans_start = 3 if rotation_first else 0
        for i in range(N):
            vec   = groundstate_group[i]
            Phi_0 = vec[rot_start:rot_start+3]
            s     = vec[trans_start:trans_start+3]
            H_inv = _splittransform_group2algebra_sv(Phi_0)
            sid = 6 * i
            rfr = sid + rot_start;  rto = rfr + 3
            tfr = sid + trans_start; tto = tfr + 3
            HX_inv[rfr:rto, rfr:rto] = H_inv
            if translation_as_midstep[i]:
                sqrtS             = _euler2rotmat_sv(0.5 * Phi_0)
                zeta_0            = sqrtS.T @ s
                zeta_0_hat_transp = _hat_map_sv(-zeta_0)
                H_half            = _splittransform_algebra2group_sv(0.5 * Phi_0)
                HX_inv[tfr:tto, rfr:rto] = -0.5 * zeta_0_hat_transp @ H_half @ H_inv
                HX_inv[tfr:tto, tfr:tto] = sqrtS
            else:
                HX_inv[tfr:tto, tfr:tto] = _euler2rotmat_sv(Phi_0)
    return HX_inv


@cond_jit(nopython=True, cache=True)
def _algebra2group_stiffmat_jit(
    groundstate_algebra: np.ndarray,   # always 2D, (N, 3) or (N, 6)
    stiff_algebra: np.ndarray,
    translations_included: bool,
    rotation_first: bool,
    translation_as_midstep: np.ndarray,  # np.bool_ array of length N
) -> np.ndarray:
    HX = _algebra2group_lintrans_jit(
        groundstate_algebra, translations_included, rotation_first, translation_as_midstep)
    HX_inv = np.linalg.inv(HX)
    return HX_inv.T @ stiff_algebra @ HX_inv


@cond_jit(nopython=True, cache=True)
def _group2algebra_stiffmat_jit(
    groundstate_group: np.ndarray,   # always 2D, (N, 3) or (N, 6)
    stiff_group: np.ndarray,
    translations_included: bool,
    rotation_first: bool,
    translation_as_midstep: np.ndarray,  # np.bool_ array of length N
) -> np.ndarray:
    HX_inv = _group2algebra_lintrans_jit(
        groundstate_group, translations_included, rotation_first, translation_as_midstep)
    HX = np.linalg.inv(HX_inv)
    return HX.T @ stiff_group @ HX


@cond_jit(nopython=True, cache=True)
def _algebra2group_params_jit(
    groundstate_algebra: np.ndarray,   # always 2D, (N, 3) or (N, 6)
    stiff_algebra: np.ndarray,
    translations_included: bool,
    rotation_first: bool,
    translation_as_midstep: np.ndarray,  # np.bool_ array of length N
) -> tuple:
    N = groundstate_algebra.shape[0]
    groundstate_group = np.copy(groundstate_algebra)
    if translations_included:
        rot_start   = 0 if rotation_first else 3
        trans_start = 3 if rotation_first else 0
        for i in range(N):
            if translation_as_midstep[i]:
                vrot   = groundstate_algebra[i, rot_start:rot_start+3]
                vtrans = groundstate_algebra[i, trans_start:trans_start+3]
                groundstate_group[i, trans_start:trans_start+3] = _euler2rotmat_sv(0.5 * vrot) @ vtrans
    HX_inv = _group2algebra_lintrans_jit(
        groundstate_group, translations_included, rotation_first, translation_as_midstep)
    stiff_group = HX_inv.T @ stiff_algebra @ HX_inv
    return groundstate_group, stiff_group


@cond_jit(nopython=True, cache=True)
def _group2algebra_params_jit(
    groundstate_group: np.ndarray,   # always 2D, (N, 3) or (N, 6)
    stiff_group: np.ndarray,
    translations_included: bool,
    rotation_first: bool,
    translation_as_midstep: np.ndarray,  # np.bool_ array of length N
) -> tuple:
    N = groundstate_group.shape[0]
    groundstate_algebra = np.copy(groundstate_group)
    if translations_included:
        rot_start   = 0 if rotation_first else 3
        trans_start = 3 if rotation_first else 0
        for i in range(N):
            if translation_as_midstep[i]:
                vrot   = groundstate_group[i, rot_start:rot_start+3]
                vtrans = groundstate_group[i, trans_start:trans_start+3]
                groundstate_algebra[i, trans_start:trans_start+3] = _euler2rotmat_sv(0.5 * vrot).T @ vtrans
    HX = _algebra2group_lintrans_jit(
        groundstate_algebra, translations_included, rotation_first, translation_as_midstep)
    stiff_algebra = HX.T @ stiff_group @ HX
    return groundstate_algebra, stiff_algebra


##########################################################################################################
########## Linear transformations between algebra and group definition of fluctuations ###################
##########################################################################################################

def algebra2group_lintrans(
    groundstate_algebra: np.ndarray,  # shape (N, 3) or (N, 6): groundstate in algebra splitting
    rotation_first: bool = True,
    translation_as_midstep: bool | list[bool] = False,
    optimized: bool = True
    ) -> np.ndarray:  # shape (N*3, N*3) or (N*6, N*6): linear transformation matrix
    """Linearization of the transformation of dynamic components from algebra (vector) to group (matrix) splitting between static and dynamic parts. Optionally the transformations from midstep triad definition to triad definition of the translational component may also be included. 

    Args:
        groundstate_algebra (np.ndarray): 
                set of groundstate Euler vectors (Nx3 or Nx6) around which the transformation is 
                linearly expanded. If the vectors are 6-vectors translations are assumed to be included
                
        rotation_first (bool): 
                If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational 
                degrees of freedom
                
        translation_as_midstep (bool | list[bool]): 
                If True (or a list of True values), the translational component of the initial state 
                vectors and the groundstate are assumed to be defined in the midstep triad frame. The 
                translational component of resulting vectors will be defined in the standard SE3 
                definition assuming that the splitting between static and dynamic compoent occures at 
                the level of the group (SE3): g=s*d. If a list is provided it must have N elements 
                (one per groundstate entry); a single bool is broadcast to all N entries.
                    / R   v \   / S   s \  / D   d \     / SD  Sd+s \
                g =           =                       = 
                    \ 0   1 /   \ 0   1 /  \ 0   1 /     \ 0     1  /
                
    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    if groundstate_algebra.shape[-1] == 3:
        translations_included = False
    elif groundstate_algebra.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {groundstate_algebra.shape}")

    if len(groundstate_algebra.shape) > 2:
        raise ValueError(f'Unexpected shape of groundstate_algebra {groundstate_algebra.shape}')
    if len(groundstate_algebra.shape) == 1:
        groundstate_algebra = np.array([groundstate_algebra])

    N = len(groundstate_algebra)
    if isinstance(translation_as_midstep, bool):
        translation_as_midstep = [translation_as_midstep] * N
    elif len(translation_as_midstep) != N:
        raise ValueError(
            f"translation_as_midstep has {len(translation_as_midstep)} elements but "
            f"groundstate_algebra has {N} entries."
        )

    if optimized:
        return _algebra2group_lintrans_jit(
            groundstate_algebra, translations_included, rotation_first,
            np.array(translation_as_midstep, dtype=np.bool_))

    # define index selection
    if translations_included and rotation_first:
        rot_from = 0
        rot_to   = 3
        trans_from = 3
        trans_to   = 6
        rotslice   = slice(0,3)
        transslice = slice(3,6)
    else:
        rot_from = 3
        rot_to   = 6
        trans_from = 0
        trans_to   = 3
        transslice = slice(0,3)
        rotslice   = slice(3,6)
    
    # initialize transformation matrix
    dim = len(groundstate_algebra)*3 
    if translations_included:
        dim *= 2  
    HX = np.eye(dim)
    
    if not translations_included:
        for i, vec in enumerate(groundstate_algebra):
            HX[
                3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)
            ] = splittransform_algebra2group_batch(vec)
    else:
        for i, vec in enumerate(groundstate_algebra):
            Omega_0 = vec[rotslice]
            zeta_0  = vec[transslice]
            H = splittransform_algebra2group_batch(Omega_0)
            sid = 6*i
            rfr = sid+rot_from
            rto = sid+rot_to
            tfr = sid+trans_from
            tto = sid+trans_to
            HX[rfr:rto,rfr:rto] = H
            
            if translation_as_midstep[i]:
                sqrtS_transp = euler2rotmat_batch(-0.5*Omega_0)
                zeta_0_hat_transp = hat_map_batch(-zeta_0)    
                
                # HX[tfr:tto,rfr:rto] = 0.5 * sqrtS_transp @ zeta_0_hat_transp @ H
                H_half = splittransform_algebra2group_batch(0.5*Omega_0)
                HX[tfr:tto,rfr:rto] = 0.5 * sqrtS_transp @ zeta_0_hat_transp @ H_half
                
                HX[tfr:tto,tfr:tto] = sqrtS_transp
            else:
                ST = euler2rotmat_batch(-Omega_0)
                HX[tfr:tto,tfr:tto] = ST
                
    return HX


def group2algebra_lintrans(
    groundstate_group: np.ndarray,  # shape (N, 3) or (N, 6): groundstate in group splitting
    rotation_first: bool = True,
    translation_as_midstep: bool | list[bool] = False,
    optimized: bool = True
    ) -> np.ndarray:  # shape (N*3, N*3) or (N*6, N*6): linear transformation matrix
    """Linearization of the transformation of dynamic components from group (matrix) to algebra (vector) splitting between static and dynamic parts. Optionally the translational component may be expressed in terms of the midstep triad. 

    Args:
        groundstate_group (np.ndarray): 
                set of groundstate Euler vectors (Nx3 or Nx6) around which the transformation is 
                linearly expanded. If the vectors are 6-vectors translations are assumed to be included
                
        rotation_first (bool): 
                If the vectors are 6-vectors, the first 3 coordinates are taken to be the rotational 
                degrees of freedom
        
        translation_as_midstep (bool | list[bool]): 
                If True (or a list of True values), the translational component of the final state 
                will be expressed in terms of the midstep triad. If a list is provided it must have 
                N elements (one per groundstate entry); a single bool is broadcast to all N entries.
        
    Returns:
        float: Linear transformation matrix that transforms small deviations around the given groundstate
    """
    if groundstate_group.shape[-1] == 3:
        translations_included = False
    elif groundstate_group.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors (if translations are included). Instead received shape {groundstate_group.shape}")

    if len(groundstate_group.shape) > 2:
        raise ValueError(f'Unexpected shape of groundstate_group {groundstate_group.shape}')
    if len(groundstate_group.shape) == 1:
        groundstate_group = np.array([groundstate_group])

    N = len(groundstate_group)
    if isinstance(translation_as_midstep, bool):
        translation_as_midstep = [translation_as_midstep] * N
    elif len(translation_as_midstep) != N:
        raise ValueError(
            f"translation_as_midstep has {len(translation_as_midstep)} elements but "
            f"groundstate_group has {N} entries."
        )

    if optimized:
        return _group2algebra_lintrans_jit(
            groundstate_group, translations_included, rotation_first,
            np.array(translation_as_midstep, dtype=np.bool_))

    # define index selection
    if translations_included and rotation_first:
        rot_from = 0
        rot_to   = 3
        trans_from = 3
        trans_to   = 6
        rotslice   = slice(0,3)
        transslice = slice(3,6)
    else:
        rot_from = 3
        rot_to   = 6
        trans_from = 0
        trans_to   = 3
        transslice = slice(0,3)
        rotslice   = slice(3,6)
    
    # initialize transformation matrix
    dim = len(groundstate_group)*3 
    if translations_included:
        dim *= 2  
    HX_inv = np.eye(dim)
    
    if not translations_included:
        for i, vec in enumerate(groundstate_group):
            HX_inv[
                3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)
            ] = splittransform_group2algebra_batch(vec)
    else:
        for i, vec in enumerate(groundstate_group):
            # Phi_0 = Omega_0
            Phi_0 = vec[rotslice]
            s     = vec[transslice]
            
            H_inv = splittransform_group2algebra_batch(Phi_0)
            sid = 6*i
            rfr = sid+rot_from
            rto = sid+rot_to
            tfr = sid+trans_from
            tto = sid+trans_to
            
            HX_inv[rfr:rto,rfr:rto] = H_inv
            
            if translation_as_midstep[i]:
                sqrtS = euler2rotmat_batch(0.5*Phi_0)
                zeta_0 = sqrtS.T @ s
                zeta_0_hat_transp = hat_map_batch(-zeta_0)    
                
                # HX_inv[tfr:tto,rfr:rto] = -0.5 * zeta_0_hat_transp
                H_half = splittransform_algebra2group_batch(0.5*Phi_0)
                HX_inv[tfr:tto,rfr:rto] = -0.5 * zeta_0_hat_transp @ H_half @ H_inv
                
                HX_inv[tfr:tto,tfr:tto] = sqrtS
            else:
                S = euler2rotmat_batch(Phi_0)
                HX_inv[tfr:tto,tfr:tto] = S
                
    return HX_inv


##########################################################################################################
##########################################################################################################
############### Convert stiffnessmatrix between different definitions of rotation DOFS ###################
##########################################################################################################
##########################################################################################################

def algebra2group_stiffmat(
    groundstate_algebra: np.ndarray,  # shape (N, 3) or (N, 6): groundstate in algebra splitting
    stiff_algebra: np.ndarray,  # shape (N*ndims, N*ndims): stiffness matrix in algebra definition
    rotation_first: bool = True,
    translation_as_midstep: bool | list[bool] = False,
    optimized: bool = True
    ) -> np.ndarray:  # shape (N*ndims, N*ndims): stiffness matrix in group definition
    """Converts stiffness matrix from algebra-level (vector) splitting between static 
    and dynamic component to group-level (matrix) splitting. Optionally, the transformations 
    from midstep triad definition to triad definition of the translational component may 
    also be included.  

    Args:
        groundstate_algebra (np.ndarray): 
                groundstate expressed in algebra (vector) splitting definition. The rotational 
                component is the same in the algebra and group definition.
        
        stiff_algebra (np.ndarray): 
                stiffness matrix expressed in arbitrary units
        
        rotation_first (bool): 
                If the vectors are 6-vectors, the first 3 coordinates are taken to be the 
                rotational degrees of freedom
        
        translation_as_midstep (bool | list[bool]): 
                If True (or a list of True values), the translational component of the initial 
                state stiffness matrix and the groundstate are assumed to be defined in the 
                midstep triad frame. The translational component of resulting vectors will be 
                defined in the standard SE3 definition assuming that the splitting between 
                static and dynamic compoent occures at the level of the group (SE3): g=s*d. 
                If a list is provided it must have N elements; a single bool is broadcast to 
                all N entries.

    Returns:
        np.ndarray: Transformed stiffness matrix.
    """

    if optimized:
        if groundstate_algebra.shape[-1] == 3:
            translations_included = False
        elif groundstate_algebra.shape[-1] == 6:
            translations_included = True
        else:
            raise ValueError(f"Expected set of 3-vectors or 6-vectors. Instead received shape {groundstate_algebra.shape}")
        if len(groundstate_algebra.shape) > 2:
            raise ValueError(f'Unexpected shape of groundstate_algebra {groundstate_algebra.shape}')
        if len(groundstate_algebra.shape) == 1:
            groundstate_algebra = np.array([groundstate_algebra])
        N = len(groundstate_algebra)
        if isinstance(translation_as_midstep, bool):
            translation_as_midstep = [translation_as_midstep] * N
        elif len(translation_as_midstep) != N:
            raise ValueError(
                f"translation_as_midstep has {len(translation_as_midstep)} elements but "
                f"groundstate_algebra has {N} entries."
            )
        return _algebra2group_stiffmat_jit(
            groundstate_algebra, stiff_algebra, translations_included, rotation_first,
            np.array(translation_as_midstep, dtype=np.bool_))

    HX = algebra2group_lintrans(
        groundstate_algebra,
        rotation_first=rotation_first,
        translation_as_midstep=translation_as_midstep,
        optimized=False
    )
    HX_inv = np.linalg.inv(HX)
    stiff_group = HX_inv.T @ stiff_algebra @ HX_inv
    return stiff_group


def group2algebra_stiffmat(
    groundstate_group: np.ndarray,  # shape (N, 3) or (N, 6): groundstate in group splitting
    stiff_group: np.ndarray,  # shape (N*ndims, N*ndims): stiffness matrix in group definition
    rotation_first: bool = True,
    translation_as_midstep: bool | list[bool] = False,
    optimized: bool = True
    ) -> np.ndarray:  # shape (N*ndims, N*ndims): stiffness matrix in algebra definition
    """Converts stiffness matrix from group-level (matrix) splitting between static and dynamic component to algebra-level (vector) splitting. Optionally, the transformations from midstep triad definition to triad definition of the translational component may also be included. I.e. the final 
    definition will assume a midstep triad definition of the translational component. 

    Args:
        groundstate_group (np.ndarray): 
                groundstate expressed in group (matrix) splitting definition. The rotational 
                component is the same in the algebra and group definition.
                
        stiff_group (np.ndarray): 
                stiffness matrix expressed in arbitrary units
                
        rotation_first (bool): 
                If the vectors are 6-vectors, the first 3 coordinates are taken to be the 
                rotational degrees of freedom
                
        translation_as_midstep (bool | list[bool]): 
                If True (or a list of True values), the translational component of the final 
                state will be expressed in terms of the midstep triad. If a list is provided 
                it must have N elements; a single bool is broadcast to all N entries.    

    Returns:
        np.ndarray: Transformed stiffness matrix.
    """

    if optimized:
        if groundstate_group.shape[-1] == 3:
            translations_included = False
        elif groundstate_group.shape[-1] == 6:
            translations_included = True
        else:
            raise ValueError(f"Expected set of 3-vectors or 6-vectors. Instead received shape {groundstate_group.shape}")
        if len(groundstate_group.shape) > 2:
            raise ValueError(f'Unexpected shape of groundstate_group {groundstate_group.shape}')
        if len(groundstate_group.shape) == 1:
            groundstate_group = np.array([groundstate_group])
        N = len(groundstate_group)
        if isinstance(translation_as_midstep, bool):
            translation_as_midstep = [translation_as_midstep] * N
        elif len(translation_as_midstep) != N:
            raise ValueError(
                f"translation_as_midstep has {len(translation_as_midstep)} elements but "
                f"groundstate_group has {N} entries."
            )
        return _group2algebra_stiffmat_jit(
            groundstate_group, stiff_group, translations_included, rotation_first,
            np.array(translation_as_midstep, dtype=np.bool_))

    HX_inv = group2algebra_lintrans(
        groundstate_group,
        rotation_first=rotation_first,
        translation_as_midstep=translation_as_midstep,
        optimized=False
    )
    HX = np.linalg.inv(HX_inv)
    stiff_algebra = HX.T @ stiff_group @ HX
    return stiff_algebra


##########################################################################################################
##########################################################################################################
########## Convert stiffnessmatrix and groundstate between different definitions of rotation DOFS ########
##########################################################################################################
##########################################################################################################


def algebra2group_params(
    groundstate_algebra: np.ndarray,  # shape (N, 3) or (N, 6): groundstate in algebra splitting
    stiff_algebra: np.ndarray,  # shape (N*ndims, N*ndims): stiffness matrix in algebra definition
    rotation_first: bool = True,
    translation_as_midstep: bool | list[bool] = False,
    optimized: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:  # (groundstate_group (N,6), stiffness matrix (N*ndims, N*ndims)) in group definition
    """Converts both the groundstate and the stiffness matrix from algebra-level (vector)
    splitting to group-level (matrix) splitting. If the translational component is expressed 
    in the midstep triad frame, the groundstate is converted via midstep2triad and the linear 
    transformation is built accordingly, so that the returned groundstate and stiffness matrix 
    are both in the standard SE3 (triad) convention.

    Args:
        groundstate_algebra (np.ndarray): 
                groundstate expressed in algebra (vector) splitting definition (Nx3 or Nx6). 
                The rotational component is the same in the algebra and group definition. 
                When translation_as_midstep is True (for a given entry), the translational 
                component is expected in the midstep triad frame.
        
        stiff_algebra (np.ndarray): 
                stiffness matrix in algebra splitting, shape (N*ndims, N*ndims)
        
        rotation_first (bool): 
                If the vectors are 6-vectors, the first 3 coordinates are taken to be the 
                rotational degrees of freedom
        
        translation_as_midstep (bool | list[bool]): 
                If True (or the corresponding list entry is True), the translational component 
                of the groundstate and stiffness matrix are assumed to be defined in the midstep 
                triad frame. The returned groundstate and stiffness matrix will use the standard 
                SE3 (triad) definition. If a list is provided it must have N elements; a single 
                bool is broadcast to all N entries.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
                (groundstate_group, stiff_group) — the groundstate (Nx6) and stiffness matrix 
                (N*ndims, N*ndims) expressed in group (matrix) splitting. For entries where 
                translation_as_midstep is True, the translational component of groundstate_group 
                is in the SE3 (triad) frame.
    """

    if groundstate_algebra.shape[-1] == 3:
        translations_included = False
    elif groundstate_algebra.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors. Instead received shape {groundstate_algebra.shape}")
    if len(groundstate_algebra.shape) > 2:
        raise ValueError(f'Unexpected shape of groundstate_algebra {groundstate_algebra.shape}')
    if len(groundstate_algebra.shape) == 1:
        groundstate_algebra = np.array([groundstate_algebra])

    N = len(groundstate_algebra)
    if isinstance(translation_as_midstep, bool):
        translation_as_midstep = [translation_as_midstep] * N
    elif len(translation_as_midstep) != N:
        raise ValueError(
            f"translation_as_midstep has {len(translation_as_midstep)} elements but "
            f"groundstate_algebra has {N} entries."
        )

    if optimized:
        return _algebra2group_params_jit(
            groundstate_algebra, stiff_algebra, translations_included, rotation_first,
            np.array(translation_as_midstep, dtype=np.bool_))

    groundstate_group = np.copy(groundstate_algebra)
    for i in range(len(groundstate_group)):
        if translation_as_midstep[i]:
            groundstate_group[i] = midstep2triad(groundstate_group[i])

    HX_inv = group2algebra_lintrans(
        groundstate_group,
        rotation_first=rotation_first,
        translation_as_midstep=translation_as_midstep,
        optimized=False
    )
    stiff_group = HX_inv.T @ stiff_algebra @ HX_inv
    return groundstate_group, stiff_group


def group2algebra_params(
    groundstate_group: np.ndarray,  # shape (N, 3) or (N, 6): groundstate in group splitting
    stiff_group: np.ndarray,  # shape (N*ndims, N*ndims): stiffness matrix in group definition
    rotation_first: bool = True,
    translation_as_midstep: bool | list[bool] = False,
    optimized: bool = True
    ) -> tuple[np.ndarray, np.ndarray]:  # (groundstate_algebra (N,6), stiffness matrix (N*ndims, N*ndims)) in algebra definition
    """Converts both the groundstate and the stiffness matrix from group-level (matrix)
    splitting to algebra-level (vector) splitting. If the translational component should be 
    expressed in the midstep triad frame, the groundstate is converted via triad2midstep and 
    the linear transformation is built accordingly, so that the returned groundstate and 
    stiffness matrix are both in the midstep triad convention.

    Args:
        groundstate_group (np.ndarray): 
                groundstate expressed in group (matrix) splitting definition (Nx3 or Nx6). 
                The rotational component is the same in the algebra and group definition. 
                When translation_as_midstep is True (for a given entry), the translational 
                component is expected in the SE3 (triad) frame.
                
        stiff_group (np.ndarray): 
                stiffness matrix in group splitting, shape (N*ndims, N*ndims)
                
        rotation_first (bool): 
                If the vectors are 6-vectors, the first 3 coordinates are taken to be the 
                rotational degrees of freedom
                
        translation_as_midstep (bool | list[bool]): 
                If True (or the corresponding list entry is True), the translational component 
                of the returned groundstate and stiffness matrix will be expressed in the midstep 
                triad frame. If a list is provided it must have N elements; a single bool is 
                broadcast to all N entries.

    Returns:
        tuple[np.ndarray, np.ndarray]: 
                (groundstate_algebra, stiff_algebra) — the groundstate (Nx6) and stiffness matrix 
                (N*ndims, N*ndims) expressed in algebra (vector) splitting. For entries where 
                translation_as_midstep is True, the translational component of groundstate_algebra 
                is in the midstep triad frame.
    """

    if groundstate_group.shape[-1] == 3:
        translations_included = False
    elif groundstate_group.shape[-1] == 6:
        translations_included = True
    else:
        raise ValueError(f"Expected set of 3-vectors or 6-vectors. Instead received shape {groundstate_group.shape}")
    if len(groundstate_group.shape) > 2:
        raise ValueError(f'Unexpected shape of groundstate_group {groundstate_group.shape}')
    if len(groundstate_group.shape) == 1:
        groundstate_group = np.array([groundstate_group])

    N = len(groundstate_group)
    if isinstance(translation_as_midstep, bool):
        translation_as_midstep = [translation_as_midstep] * N
    elif len(translation_as_midstep) != N:
        raise ValueError(
            f"translation_as_midstep has {len(translation_as_midstep)} elements but "
            f"groundstate_group has {N} entries."
        )

    if optimized:
        return _group2algebra_params_jit(
            groundstate_group, stiff_group, translations_included, rotation_first,
            np.array(translation_as_midstep, dtype=np.bool_))

    groundstate_algebra = np.copy(groundstate_group)
    for i in range(len(groundstate_algebra)):
        if translation_as_midstep[i]:
            groundstate_algebra[i] = triad2midstep(groundstate_algebra[i])

    HX = algebra2group_lintrans(
        groundstate_algebra,
        rotation_first=rotation_first,
        translation_as_midstep=translation_as_midstep,
        optimized=False
    )
    stiff_algebra = HX.T @ stiff_group @ HX
    return groundstate_algebra, stiff_algebra



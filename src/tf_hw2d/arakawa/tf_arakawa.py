"""
PhiML-based Arakawa Scheme Implementations
==========================================

This module provides implementations of the Arakawa scheme for computing the Poisson bracket using the PhiML library. 

The module includes:

- A wrapper for PhiML's periodic 2d arakawa poisson bracket.

Note: The function in this module require the vorticity field (`zeta`), the stream function field (`psi`) and the dx scalar as primary inputs.
"""


from phiml import math

@math.jit_compile
def periodic_arakawa_vec(zeta, psi, dx):
    """
    Compute the Poisson bracket (Jacobian) of vorticity and streamfunction for a 2D periodic
    domain using a vectorized version of the Arakawa scheme. This function automatically
    handles the required padding.

    Args:
        zeta (Tensor): Vorticity field.
        psi (Tensor): Stream function field.
        dx (float): Grid spacing.

    Returns:
        Tensor: Discretized Poisson bracket (Jacobian) over the grid without padding.
    """
    return math._nd._periodic_2d_arakawa_poisson_bracket(zeta, psi, dx)

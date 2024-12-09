"""
Poisson Solvers with PhiML
========================================

This module provides a set of functions to solve the Poisson equation using the Conjugate Gradient or Fourier transform approach with the PhiML library. The Fourier method is especially beneficial for periodic domains and spectral methods, as it can retrieve the original field from its gradient or Laplace efficiently in the spectral space.

Main functionalities include:

- `fourier_poisson_double`: Solves the Poisson equation withthe Fourier transform and PhiML default precision.
- `CG_poisson`: Solves the Poisson equation with a compact 4th finite difference scheme.

These functions are designed for both prototyping and production-level simulations, offering a balance between accuracy and performance. They are particularly well-suited for large-scale simulations in periodic domains.
"""

import numpy as np
from phiml import math


@math.jit_compile
def fourier_poisson_double(
    tensor: math.Tensor, dx: float, times: int = 1
) -> math.Tensor:
    """Inverse operation to `fourier_laplace`."""
    return math.fourier_poisson(tensor, dx=dx, times=times) #IS THIS CAUSING TENSORFLOW2.15 (not with 2.10) to say Warning complex to float conversion will discard imaginary??
    # return math.real(math.fourier_poisson(tensor, dx=dx, times=times))


@math.jit_compile_linear
def laplace_2d(x: math.Tensor, deltax: float) -> math.Tensor:
    """Constructs the left-hand-side matrice A to solve the 2D poisson equation as Ax=b with a second order finite difference scheme."""
    mode = math.extrapolation.PERIODIC
    return (
        math.pad(x, {"y": (0, 1), "x": (0, 0)}, mode=mode).y[1:]
        + math.pad(x, {"y": (1, 0), "x": (0, 0)}, mode=mode).y[:-1]
        + math.pad(x, {"y": (0, 0), "x": (0, 1)}, mode=mode).x[1:]
        + math.pad(x, {"y": (0, 0), "x": (1, 0)}, mode=mode).x[:-1]
        - 4 * x
    ) / (deltax * deltax)


@math.jit_compile_linear
def laplace_2d_compact(x: math.Tensor, deltax: float) -> math.Tensor:
    """Constructs the left-hand-side matrice A to solve the 2D poisson equation as Ax=b with a 4th order compact finite difference scheme."""
    mode = math.extrapolation.PERIODIC
    return (
        (1.0 / 6.0)
        * (
            4.0
            * (
                math.pad(x, {"y": (0, 1), "x": (0, 0)}, mode=mode).y[1:]
                + math.pad(x, {"y": (1, 0), "x": (0, 0)}, mode=mode).y[:-1]
                + math.pad(x, {"y": (0, 0), "x": (0, 1)}, mode=mode).x[1:]
                + math.pad(x, {"y": (0, 0), "x": (1, 0)}, mode=mode).x[:-1]
                - 5.0 * x
            )
            + (
                math.pad(x, {"y": (0, 1), "x": (0, 1)}, mode=mode).x[1:].y[1:]
                + math.pad(x, {"y": (1, 0), "x": (0, 1)}, mode=mode).x[1:].y[:-1]
                + math.pad(x, {"y": (0, 1), "x": (1, 0)}, mode=mode).x[:-1].y[1:]
                + math.pad(x, {"y": (1, 0), "x": (1, 0)}, mode=mode).x[:-1].y[:-1]
            )
        )
        / (deltax * deltax)
    )


def CG_poisson(tensor: math.Tensor, dx: float, x0: math.Tensor) -> math.Tensor:
    """Solves de Poisson equation using CG.
    tensor is the RHS.
    x0 is the initial guess.
    """
    solve = math.Solve(
        "auto",
        rel_tol=1e-5,
        abs_tol=1e-5,
        x0=x0 - math.mean(x0) - 1.0,
        preconditioner=None,
        max_iterations=100,
        suppress=(
            math.ConvergenceException,
            math._optimize.NotConverged,
        ),
    )  # WORKS!!!! IF MEAN OF X0 IS DIFFERENT THAN 0.0 with CG IT CONVERGES TO A SOLUTION WITH THAT MEAN!!
    # sol = math.solve_linear(math.jit_compile_linear(laplace_2d), y=tensor, solve=solve, deltax=dx)
    # sol = math.solve_linear(laplace_2d, y=tensor, solve=solve, deltax=dx)
    tensor_convolve = math.convolve(
        tensor,
        math.tensor(
            (1.0 / 12.0)
            * np.array([[0.0, 1.0, 0.0], [1.0, 8.0, 1.0], [0.0, 1.0, 0.0]]),
            math.spatial("y", "x"),
        ),
        extrapolation=math.extrapolation.PERIODIC,
    )
    sol = math.solve_linear(
        laplace_2d_compact,
        y=tensor_convolve - math.mean(tensor_convolve),
        solve=solve,
        deltax=dx,
    )
    sol = sol - math.mean(sol)
    return sol

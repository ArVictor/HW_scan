"""
Numerical Properties using PhiML (`tf_properties`)
=====================================================

This module provides a collection of functions to compute various properties and metrics related to a 2D Hasegawa-Wakatani system.
It leverages the PhiML library for efficient computations on grid-based data.
The provided functionalities help in understanding the physical and spectral properties of the system.

Specifically, the module includes:

- **Sources and Sinks** such as $\Gamma_n$ and $Gamma_c$.
- **Energies** including total, kinetic, and potential energy.
- **Enstrophy** to quantify the system's vorticity content.
- **Dissipation Metrics** to understand the system's energy dissipation rate over time.
- **Spectral Properties** of various metrics for further analysis and verification.

Refer to each function's docstring for detailed information on their use and mathematical formulation.
"""
# Define Frame Properties
# Assume (..., y, x) as shape
from typing import Tuple, Union
from tf_hw2d.gradients.tf_gradients import periodic_laplace_N, gradient
from phiml import math


# Gammas
@math.jit_compile
def get_gamma_n(
    n: math.Tensor, p: math.Tensor, dx: float, dy_p=None
) -> Union[float, math.Tensor]:
    """
    Compute the average particle flux ($\Gamma_n$) using the formula:
    $$
        \Gamma_n = - \int{d^2 x \tilde{n} \frac{\partial \tilde{\phi}}{\partial y}}
    $$

    Args:
        n (math.Tensor): Density (or similar field).
        p (math.Tensor): Potential (or similar field).
        dx (float): Grid spacing.
        dy_p (math.Tensor, optional): Gradient of potential in the y-direction.
            Computed from `p` if not provided.

    Returns:
        float: Computed average particle flux value.
    """
    if dy_p is None:
        dy_p = gradient(p, dx=dx, scheme="mixed_all", order=20, axis="y")
    gamma_n = -math.mean((n * dy_p), dim=math.spatial("y", "x"))
    print("gamma_n:", gamma_n)
    return gamma_n


@math.jit_compile
def get_gamma_c(
    n: math.Tensor, p: math.Tensor, c1: float, dx: float
) -> Union[float, math.Tensor]:
    """
    Compute the sink $\Gamma_c$ using the formula:
    $$
        \Gamma_c = c_1 \int{d^2 x (\tilde{n} - \tilde{\phi})^2}
    $$

    Args:
        n (math.Tensor): Density (or similar field).
        p (math.Tensor): Potential (or similar field).
        c1 (float): Proportionality constant.
        dx (float): Grid spacing.

    Returns:
        float: Computed particle flux value.
    """

    gamma_c = c1 * math.mean((n - p) ** 2)
    return gamma_c


# Spectral Gamma_n


@math.jit_compile
def get_gamma_n_ky(
    n: math.Tensor, p: math.Tensor, dx: float
) -> Tuple[math.Tensor, math.Tensor]:
    """Calculate the spectral components of Gamma_n"""
    n_dft = math.fft(n) / math.sqrt(
        math.prod(n.shape.spatial.sizes)
    )  # The division is to obtain numpy "ortho" fft2
    p_dft = math.fft(p) / math.sqrt(
        math.prod(p.shape.spatial.sizes)
    )  # The division is to obtain numpy "ortho" fft2
    k_ky = math.fftfreq(n.shape["y"], dx=dx) * 2 * math.pi
    k_ky = math.slice(k_ky, slices={"vector": "y"})
    gamma_n_k = n_dft * 1j * k_ky * math.conjugate(p_dft)  # gamma_n(ky, kx)
    integrated_gamma_n_k = math.mean(
        math.real(gamma_n_k),
        dim=math.spatial("x"),
    )  # gamma_n(ky)
    integrated_gamma_n_k = math.boolean_mask(
        integrated_gamma_n_k,
        dim=math.spatial("y"),
        mask=k_ky >= 0,
    )
    ky = math.slice(k_ky, slices={"vector": "y"})
    ky = math.boolean_mask(ky, dim=math.spatial("y"), mask=k_ky >= 0)
    return ky, integrated_gamma_n_k


@math.jit_compile
def get_gamma_n_spectrally(
    n: math.Tensor, p: math.Tensor, dx: float
) -> Union[float, math.Tensor]:
    # DOESN'T WORK!!!
    ky, integrated_gamma_n_k = get_gamma_n_ky(n=n, p=p, dx=dx)
    gamma_n = math.mean(integrated_gamma_n_k, dim=math.spatial("y"))  # gamma_n
    return gamma_n


# Energy


@math.jit_compile
def get_energy(n: math.Tensor, phi: math.Tensor, dx: float) -> math.Tensor:
    """Energy of the HW2D system, sum of thermal and kinetic energy
    $$ E = \\frac{1}{2} \int{d^2 x \left(n^2 + | \nabla \phi|^2 \right)} $$
    """
    grad_phi = (
        gradient(phi, dx=dx, scheme="mixed_all", order=20, axis="x") ** 2
        + gradient(phi, dx=dx, scheme="mixed_all", order=20, axis="y") ** 2
    )
    # Norm
    norm_grad_phi = grad_phi  # math.abs(grad_phi)
    # Integrate, then divide by 2
    integral = math.mean((n**2) + norm_grad_phi)
    return integral / 2


# Thermal Energy


@math.jit_compile
def get_thermal_energy(n: math.Tensor) -> math.Tensor:
    """Energy of the HW2D system, sum of thermal and kinetic energy
    $$ E = \\frac{1}{2} \int{d^2 x \left(n^2 \right)} $$
    """
    return 0.5 * math.mean((n**2))  # Mean instead of integral


# Kinetic Energy


@math.jit_compile
def get_kinetic_energy_robin(phi: math.Tensor, dx: float) -> math.Tensor:
    """Energy of the HW2D system, sum of thermal and kinetic energy
    $$ E = \\frac{1}{2} \int{d^2 x \left(| \nabla \phi|^2 \right)} $$
    """
    grad_phi = gradient(phi, dx=dx, scheme="mixed_all", order=20, axis="x") + gradient(
        phi, dx=dx, scheme="mixed_all", order=20, axis="y"
    )
    # Norm
    norm_grad_phi = grad_phi  # math.abs(grad_phi)
    # Integrate, then divide by 2
    integral = math.mean((norm_grad_phi**2))  # Mean instead of integral
    return integral / 2


# Kinetic Energy


@math.jit_compile
def get_kinetic_energy_victor(phi: math.Tensor, dx: float) -> math.Tensor:
    """Energy of the HW2D system, sum of thermal and kinetic energy
    $$ E = \\frac{1}{2} \int{d^2 x \left(| \nabla \phi|^2 \right)} $$
    """
    # L2 norm
    norm_grad_phi = (
        gradient(phi, dx=dx, scheme="mixed_all", order=20, axis="x") ** 2
        + gradient(phi, dx=dx, scheme="mixed_all", order=20, axis="y") ** 2
    )
    # No sqrt since the norm is squared
    # Integrate, then divide by 2
    integral = math.mean(norm_grad_phi)  # Mean instead of integral
    return integral / 2


# Enstrophy


@math.jit_compile
def get_enstrophy(n: math.Tensor, omega: math.Tensor, dx: float) -> math.Tensor:
    """Enstrophy of the HW2D system
    $$
        U = \frac{1}{2} \int{d^2 x (n^2 - \nabla^2 \phi)^2}
          = \frac{1}{2} \int{d^2 x (n-\Omega)^2}
    $$
    """
    integral = math.mean(((n - omega) ** 2))
    return integral / 2


@math.jit_compile
def get_enstrophy_phi(n: math.Tensor, phi: math.Tensor, dx: float) -> math.Tensor:
    """Enstrophy of the HW2D system from phi
    $$
        U = \frac{1}{2} \int{d^2 x (n^2 - \nabla^2 \phi)^2}
          = \frac{1}{2} \int{d^2 x (n-\Omega)^2}
    $$
    """
    # omega = math.laplace(phi, dx=dx, padding='periodic')
    omega = periodic_laplace_N(phi, dx=dx, N=1)
    omega -= math.mean(omega)
    integral = math.mean(((n - omega) ** 2))
    return integral / 2


@math.jit_compile
def get_D(arr: math.Tensor, nu: float, N: int, dx: float) -> math.Tensor:
    D = periodic_laplace_N(arr, dx=dx, N=N)
    return nu * D


# Sinks


@math.jit_compile
def get_DE(
    n: math.Tensor, p: math.Tensor, Dn: math.Tensor, Dp: math.Tensor
) -> Union[float, math.Tensor]:
    DE = math.mean(n * Dn - p * Dp)
    return DE


@math.jit_compile
def get_DU(
    n: math.Tensor, o: math.Tensor, Dn: math.Tensor, Dp: math.Tensor
) -> Union[float, math.Tensor]:
    DE = -math.mean((n - o) * (Dn - Dp))
    return DE


# Time Variation


@math.jit_compile
def get_dE_dt(
    gamma_n: math.Tensor, gamma_c: math.Tensor, DE: math.Tensor
) -> Union[float, math.Tensor]:
    return gamma_n - gamma_c - DE


@math.jit_compile
def get_dU_dt(gamma_n: math.Tensor, DU: math.Tensor) -> Union[float, math.Tensor]:
    return gamma_n - DU


# Spectral Energies


@math.jit_compile
def get_energy_N_ky(n: math.Tensor) -> math.Tensor:
    """thermal energy
    $$ E^N(k) = \frac{1}{2} |n(k)|^2 $$
    """
    n_dft = math.fft(n) / math.sqrt(
        math.prod(n.shape.spatial.sizes)
    )  # The division is to obtain numpy "ortho" fft2
    E_N_ky = math.abs(n_dft) ** 2 / 2
    E_N_ky = math.mean(E_N_ky, dim=math.spatial("x"))
    return E_N_ky


@math.jit_compile
def get_energy_N_spectrally(n: math.Tensor) -> math.Tensor:
    E_N_ky = get_energy_N_ky(n)
    E_N = math.mean(E_N_ky, dim=math.spatial("y"))
    return E_N


@math.jit_compile
def get_energy_V_ky(p: math.Tensor, dx: float) -> math.Tensor:
    """kinetic energy
    $$ E^V(k) = \frac{1}{2} |k \phi(k) |^2 $$
    """
    p_dft = math.fft(p) / math.sqrt(
        math.prod(p.shape.spatial.sizes)
    )  # The division is to obtain numpy "ortho" fft2
    k_times_p_dft = math.sum(
        [math.fftfreq(i, dx=dx)[0] * 2 * math.pi * p_dft for i in p.shape.spatial],
        dim="0",
    )
    E_V_ky = math.abs(k_times_p_dft) ** 2 / 2
    E_V_ky = math.mean(E_V_ky, dim=math.spatial("x"))
    return E_V_ky


@math.jit_compile
def get_energy_V_spectrally(p: math.Tensor, dx: float) -> math.Tensor:
    E_V_ky = get_energy_V_ky(p, dx=dx)
    E_V = math.mean(E_V_ky, dim=math.spatial("y"))
    return E_V


@math.jit_compile
def get_energy_kinetic_ky_victor(p: math.Tensor, dx: float) -> math.Tensor:
    """kinetic energy
    $$ E^V(k) = \frac{1}{2} |k \phi(k) |^2 $$
    """
    p_dft = math.fft(p) / math.sqrt(
        math.prod(p.shape.spatial.sizes)
    )  # The division is to obtain numpy "ortho" fft2
    k_times_p_dft = math.sum(
        [
            ((math.fftfreq(i, dx=dx)[0] * 2 * math.pi) ** 2)
            * p_dft
            * math.conjugate(p_dft)
            for i in p.shape.spatial
        ],
        dim="0",
    )
    E_V_ky = math.mean(k_times_p_dft, dim=math.spatial("x"))
    E_V_ky = E_V_ky / 2
    return E_V_ky


@math.jit_compile
def get_energy_kinetic_spectrally_victor(p: math.Tensor, dx: float) -> math.Tensor:
    E_V_ky = get_energy_kinetic_ky_victor(p, dx=dx)
    E_V = math.mean(E_V_ky, dim=math.spatial("y"))
    E_V = math.real(E_V)
    return E_V


@math.jit_compile
def get_energy_kinetic_ky_victor_PIoutside(p: math.Tensor, dx: float) -> math.Tensor:
    """kinetic energy
    $$ E^V(k) = \frac{1}{2} |k \phi(k) |^2 $$
    """
    p_dft = math.fft(p) / math.sqrt(
        math.prod(p.shape.spatial.sizes)
    )  # The division is to obtain numpy "ortho" fft2
    k_times_p_dft = math.sum(
        [
            2
            * math.pi
            * ((math.fftfreq(i, dx=dx)[0]) ** 2)
            * p_dft
            * math.conjugate(p_dft)
            for i in p.shape.spatial
        ],
        dim="0",
    )
    E_V_ky = math.mean(k_times_p_dft, dim=math.spatial("x"))
    E_V_ky = E_V_ky / 2
    return E_V_ky


@math.jit_compile
def get_energy_kinetic_spectrally_victor_PIoutside(
    p: math.Tensor, dx: float
) -> math.Tensor:
    E_V_ky = get_energy_kinetic_ky_victor_PIoutside(p, dx=dx)
    E_V = math.mean(E_V_ky, dim=math.spatial("y"))
    E_V = math.real(E_V)
    return E_V


# Phase Angle Spectra


# tf_hw2d
@math.jit_compile
def get_delta_ky(n: math.Tensor, p: math.Tensor, real=True) -> math.Tensor:
    n_dft = math.fft(n) / math.sqrt(
        math.prod(n.shape.spatial.sizes)
    )  # This is norm="ortho"
    p_dft = math.fft(p) / math.sqrt(
        math.prod(p.shape.spatial.sizes)
    )  # This is norm="ortho"
    delta_k = math.imag(math.log(math.conjugate(n_dft) * p_dft))
    delta_k = math.mean(
        delta_k, dim=math.spatial("y")
    )  # mean in x #<-that is a comment from robin. I don't know why but to get the same rsults we have to average over spatial('y').
    # Get Real component
    k_ky = math.fftfreq(
        n.shape["x"], dx=1
    )  # dx size doesn't matter, just some positive float. We compute these values to have the positive and negative components as mask
    delta_k = math.boolean_mask(delta_k, dim=math.spatial("x"), mask=k_ky >= 0)
    return delta_k

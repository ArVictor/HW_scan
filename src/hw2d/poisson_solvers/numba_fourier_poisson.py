import numpy as np
import numba

precision = np.float64
complex_precision = np.complex128

# import cupy as cp

# def fourier_poisson_double(tensor: np.ndarray, dx: float, times: int = 1) -> np.ndarray:
#     """Inverse operation to `fourier_laplace`."""
#     tensor = cp.array(tensor, dtype=cp.complex128)
#     frequencies = cp.fft.fft2(tensor)
#     result = cp.empty_like(tensor)
#     for i in range(tensor.shape[0]):
#         k = cp.meshgrid(*[cp.fft.fftfreq(int(n)) for n in tensor.shape[1:]], indexing="ij")
#         k = cp.stack(k, -1)
#         k = cp.sum(k**2, axis=-1)
#         fft_laplace = -((2 * cp.pi) ** 2) * k
#         fft_laplace[0, 0] = cp.inf
#         # with cp.errstate(divide="ignore", invalid="ignore"):
#         result[i,...] = frequencies[i, ...] / (cp.where(fft_laplace == 0, 1.0, fft_laplace)**times)
#         result[i,...] = cp.where(fft_laplace == 0, 0, result[i,...])
#     result = cp.real(cp.fft.ifft2(result))
#     return cp.asnumpy((result * dx**2).astype(cp.float64))



# @numba.jit(nopython=True, parallel=True)
def fourier_poisson_double(tensor: np.ndarray, dx: float, times: int = 1) -> np.ndarray:
    """Inverse operation to `fourier_laplace`."""
    tensor = np.array(tensor, dtype=complex_precision)
    frequencies = np.fft.fft2(tensor)

    #result = np.empty_like(tensor)
    #result_comp = np.empty_like(tensor)
    k = fftfreq_sq(tensor.shape[-2:])   #THIS IS THE ONLY MODIFICATION FOR A BATCHED COMPUTATION ?????
    # for i in numba.prange(tensor.shape[0]):
    #     result_comp[i,...] = core_computation(frequencies[i,...], k, dx, times)
    result_comp = core_computation(frequencies, k, dx, times)

    result = np.real(np.fft.ifft2(result_comp))
    #return result.astype(precision)
    return result


@numba.jit(nopython=True)
def core_computation(
    frequencies: np.ndarray, k: np.ndarray, dx: float, times: int = 1
) -> np.ndarray:
    fft_laplace = -((2 * np.pi) ** 2) * k
    # Avoiding the use of np.inf for now. Set to a very large number.
    fft_laplace[0, 0] = 1e14

    frequencies = np.where(fft_laplace == 0, 0, frequencies)
    result = frequencies / (fft_laplace**times)

    return result * dx**2


@numba.jit(nopython=True)
def fftfreq_sq(resolution: np.ndarray) -> np.ndarray:
    dim_x, dim_y = resolution
    freq_x = custom_fftfreq(dim_x)
    freq_y = custom_fftfreq(dim_y)

    # Equivalent to:  k_sq = freq_x**2 + freq_y**2
    k_sq = np.empty((dim_x, dim_y), dtype=precision)
    for i in numba.prange(dim_x):
        for j in numba.prange(dim_y):
            k_sq[i, j] = freq_x[i] ** 2 + freq_y[j] ** 2

    return k_sq


@numba.jit(nopython=True)
def custom_fftfreq(n: np.ndarray) -> np.ndarray:
    """Custom FFT frequency function to replicate np.fft.fftfreq for Numba."""
    results = np.empty(n, dtype=np.int16)
    N = (n - 1) // 2 + 1
    results[:N] = np.arange(0, N)
    results[N:] = np.arange(-(n // 2), 0)
    return results / n

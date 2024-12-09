import fire
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from typing import Iterable
import pywt

# Local imports
from tf_hw2d.utils.tf_io import (
    # from utils_io import (
    get_save_params,
    create_appendable_h5,
    save_to_buffered_h5,
    append_h5,
    continue_h5_file,
)

# from hw2d.utils.namespaces import Namespace
from tf_hw2d.utils.plot.movie import create_movie

# from utils_plot_movie import create_movie
from tf_hw2d.utils.tf_run_properties import calculate_properties, add_data

# from utils_run_properties import calculate_properties
from tf_hw2d.utils.plot.timetrace import plot_timetraces

# from utils_plot_timetrace import plot_timetraces
from phiml import math

from functools import partial


def linear(
    original_data: math.Tensor,
    original_x: int,
    original_y: int,
    new_x: int,
    new_y: int,
):
    """
    Linearly interpolates the input tensor original_data.

    Args:
        original_data (Math.Tensor): Orinigal data to resample.
        original_x (int, optional): Original size of x.
        original_y (int, optional): Original size of y.
        new_x (int, optional): New size of x.
        new_y (int, optional): New size of y.

    Returns:
        math.Tensor: The linearly interpolated original_data with new dimension (new_y, new_x)
    """
    # math.use('numpy')
    math.use("tensorflow")
    # Unpacking
    shift_y = (original_y-1)/new_y/2.0
    shift_x = (original_x-1)/new_x/2.0
    subsampling_coordinates = math.meshgrid(
        y=math.linspace(0+shift_y, original_y - 1-shift_y, math.spatial(y=new_y)),
        x=math.linspace(0+shift_x, original_x - 1-shift_x, math.spatial(x=new_x)),
    )
    return math.grid_sample(
        original_data,
        subsampling_coordinates,
        # extrap=math.extrapolation.NONE, #PHIML BUG??
        extrap=None,
    )


def mean(
    original_data: math.Tensor,
    original_x: int,
    original_y: int,
    new_x: int,
    new_y: int,
):
    """
    Linearly interpolates the input tensor original_data.

    Args:
        original_data (Math.Tensor): Orinigal data to resample.
        original_x (int, optional): Original size of x.
        original_y (int, optional): Original size of y.
        new_x (int, optional): New size of x.
        new_y (int, optional): New size of y.

    Returns:
        math.Tensor: The linearly interpolated original_data with new dimension (new_y, new_x)
    """
    # math.use('numpy')
    math.use("tensorflow")
    # Unpacking
    subsampling_coordinates = math.meshgrid(
        y=math.linspace(0, original_y - 1, math.spatial(y=new_y)),
        x=math.linspace(0, original_x - 1, math.spatial(x=new_x)),
    )
    return math.grid_sample(
        original_data,
        subsampling_coordinates,
        # extrap=math.extrapolation.NONE, #PHIML BUG??
        extrap=None,
    )


def fourier(
    original_data: math.Tensor,
    original_x: int,
    original_y: int,
    new_x: int,
    new_y: int,
):
    """
    Fourier resampling the input tensor original_data.

    Args:
        original_data (Math.Tensor): Orinigal data to resample.
        original_x (int, optional): Original size of x.
        original_y (int, optional): Original size of y.
        new_x (int, optional): New size of x.
        new_y (int, optional): New size of y.

    Returns:
        math.Tensor: The fourier resampled original_data with new dimension (new_y, new_x)
    """
    # math.use('numpy')
    math.use("tensorflow")
    # Unpacking
    x_fft = math.fft(original_data)
    x_fft_roll = math.shift(
        x_fft, (int(original_x // 2),), dims=math.spatial("x"), padding="periodic"
    )[0]
    x_fft_roll = math.slice(x_fft_roll, {"shift": "x"})
    x_fft_roll = math.shift(
        x_fft_roll, (int(original_y // 2),), dims=math.spatial("y"), padding="periodic"
    )[0]
    x_fft_roll = math.slice(x_fft_roll, {"shift": "y"})
    x_fft_roll_sub_x = x_fft_roll.x[
        int(original_x // 2) - (new_x // 2) : int(original_x // 2) + (new_x // 2)
    ]
    x_fft_roll_sub = x_fft_roll_sub_x.y[
        int(original_y // 2) - (new_y // 2) : int(original_y // 2) + (new_y // 2)
    ]
    x_fft_roll_sub = x_fft_roll_sub / (
        (original_x * original_y) / (new_x * new_y)
    )  # DIVIDE BY SUBSAMPLING RATIO!!!
    x_fft_roll_sub_unroll = math.shift(
        x_fft_roll_sub, (int(new_x // 2),), dims=math.spatial("x"), padding="periodic"
    )[0]
    x_fft_roll_sub_unroll = math.slice(x_fft_roll_sub_unroll, {"shift": "x"})
    x_fft_roll_sub_unroll = math.shift(
        x_fft_roll_sub_unroll,
        (int(new_y // 2),),
        dims=math.spatial("y"),
        padding="periodic",
    )[0]
    x_fft_roll_sub_unroll = math.slice(x_fft_roll_sub_unroll, {"shift": "y"})
    x_sub = math.real(math.ifft(x_fft_roll_sub_unroll))
    return x_sub


def lowpassfilter2D(signal, thresh = 0.63, wavelet="db4"):
    thresh = thresh*np.nanmax(signal)
    boundary = 'per'
    coeff = pywt.wavedec2(signal, wavelet, mode=boundary )
    coeff[1:] = [[pywt.threshold(ii, value=thresh, mode="soft" ) for ii in i] for i in coeff[1:]]
    reconstructed_signal = pywt.waverec2(coeff, wavelet, mode=boundary )
    return reconstructed_signal



def wavelet(
    original_data: math.Tensor,
    original_x: int,
    original_y: int,
    new_x: int,
    new_y: int,
):
    """
    Downsamples original_data by first applying a lowpass filter with wavelets.

    Args:
        original_data (Math.Tensor): Orinigal data to resample.
        original_x (int, optional): Original size of x.
        original_y (int, optional): Original size of y.
        new_x (int, optional): New size of x.
        new_y (int, optional): New size of y.

    Returns:
        math.Tensor: The linearly interpolated original_data with new dimension (new_y, new_x)
    """
    # math.use('numpy')
    math.use("tensorflow")
    # Unpacking
    numpy_data = original_data.numpy(original_data.shape.names)
    #numpy_sub = lowpassfilter2D(numpy_data, 1000.63, 'db' + str(int(original_y//new_y)))[..., ::original_y//new_y,::original_x//new_x]
    numpy_sub = lowpassfilter2D(numpy_data, 1000.63, 'db' + str(int(original_y//new_y)))[..., ::original_y//new_y,::original_x//new_x]
    numpy_sub_mirror = lowpassfilter2D(numpy_data[..., ::-1, ::-1], 1000.63, 'db' + str(int(original_y//new_y)))[..., ::-1, ::-1][..., ::original_y//new_y,::original_x//new_x]
    numpy_sub = 0.5*(numpy_sub+numpy_sub_mirror)
    tf_sub = math.tensor(numpy_sub, math.batch(*original_data.shape.batch.names), math.spatial(*original_data.shape.spatial.names))
    return tf_sub - math.mean(tf_sub) + math.mean(original_data)




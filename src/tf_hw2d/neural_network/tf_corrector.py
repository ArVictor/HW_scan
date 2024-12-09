"""
Corrector step with the neural network, using PhiML.
=====================================================

This module provides the corrector function for the predictor-corrector scheme.

get_corrector_function returns the callable corrector.
"""

# Define Frame Properties
# Assume (..., y, x) as shape
from typing import Callable
from phiml import math

# Local imports
from tf_hw2d.model import HW

import time

import tensorflow as tf
# Gammas


def get_corrector_function(
    mean_phi: float,
    std_omega: float,
    std_density: float,
    std_phi: float,
    hw: HW,
    parameters: dict,
) -> Callable:
    """
    Defines the corrector function and returns it.

    Args:
        mean_phi (float): Mean of phi.
        std_omega (float): Standard deviation of omega.
        std_density (float): Standard deviation of density.
        std_phi (float): Standard deviation of phi.
        hw (HW): Hasegawa-Wakatani class with a get_phi function.
        parameters (dict): Dictionary with the simulation parameters

    Returns:
        Callable: Corrector function that returns plasma.
    """

    def corrector_step(plasma, network):
        # center all fields
        # print("plasma:", plasma)
        # print("plasma.density:", plasma.density)
        plasma.density = plasma.density - math.mean(
            plasma.density, dim=math.spatial("y", "x")
        )
        plasma.omega = plasma.omega - math.mean(
            plasma.omega, dim=math.spatial("y", "x")
        )
        plasma.phi = plasma.phi - math.mean(plasma.phi, dim=math.spatial("y", "x"))
        # Standardize phi, the network input
        in_network = plasma.phi / std_phi  #ORIGINAL FROM ROBIN
        #in_network = plasma.density / std_density   #JUST TO MAKE SURE IT REALLY DOESN'T WORK WHY I GET ILLEGAL INSTRUCTION FOR DENSITY??
        #in_network = plasma.omega / std_omega   #JUST TO MAKE SURE IT REALLY DOESN'T WORK
        #print("in_network:", in_network)
        # gradx_phi = math.spatial_gradient(
        #     plasma.phi / std_phi, dims="x", dx=0.6544984694978736, difference="central", padding="periodic"
        # ).gradient[0]
        # gradx_omega = math.spatial_gradient(
        #     plasma.omega / std_omega, dims="x", dx=0.6544984694978736, difference="central", padding="periodic"
        # ).gradient[0]
        # gradx_density= math.spatial_gradient(
        #     plasma.density / std_density, dims="x", dx=0.6544984694978736, difference="central", padding="periodic"
        # ).gradient[0]
        # grady_phi = math.spatial_gradient(
        #     plasma.phi / std_phi, dims="y", dx=0.6544984694978736, difference="central", padding="periodic"
        # ).gradient[0]
        # grady_omega = math.spatial_gradient(
        #     plasma.omega / std_omega, dims="y", dx=0.6544984694978736, difference="central", padding="periodic"
        # ).gradient[0]
        # grady_density= math.spatial_gradient(
        #     plasma.density / std_density, dims="y", dx=0.6544984694978736, difference="central", padding="periodic"
        # ).gradient[0]
        # in_network = math.stack([plasma.phi / std_phi, plasma.density / std_density, plasma.omega / std_omega, gradx_phi, gradx_omega, gradx_density, grady_phi, grady_omega, grady_density], math.channel("channel"))
        #print("in_network:", in_network)
        # print(
        #     "in_network:",
        #     in_network,
        #     "in_network mean:",
        #     math.mean(in_network),
        #     "in_network std:",
        #     math.std(in_network),
        # )
        # Neural network prediction
        # channels_last = True
        # groups = (in_network.shape.batch, *in_network.shape.spatial.names, in_network.shape.channel) if channels_last else (in_network.shape.batch, in_network.shape.channel, *in_network.shape.spatial.names)
        # in_network_native = math.reshaped_native(in_network, groups, force_expand=False)
        # print("in_network_native:", in_network_native)
        # print("network(in_network_native):", network(tf.convert_to_tensor(in_network_native)))
        # print("native_call_CUSTOM(in_network):", native_call_CUSTOM(network, in_network, channels_last=True))
        t0 = time.time()
        prediction = math.native_call(network, in_network)
        total_time_native_call = time.time() - t0
        # print(
        #     "prediction mean:",
        #     math.mean(prediction),
        #     "prediction std:",
        #     math.std(prediction),
        # )
        # Unstack the two fields from the prediction
        pred_omega, pred_density = math.unstack(prediction, math.channel("vector"))
        # Center the correction
        pred_omega = pred_omega -  math.mean(pred_omega)
        pred_density = pred_density -  math.mean(pred_density)
        # Scale with the fields' standard deviations
        pred_omega = pred_omega * std_omega
        pred_density = pred_density * std_density
        # print(
        #     "pred_omega mean:",
        #     math.mean(pred_omega),
        #     "pred_density mean:",
        #     math.mean(pred_density),
        # )
        # Apply additive correction
        plasma.density = plasma.density + pred_density
        plasma.omega = plasma.omega + pred_omega
        #plasma.density = plasma.density * pred_density  #TEST MULTIPLICATIVE CORRECTION
        #plasma.omega = plasma.omega * pred_omega  #TEST MULTIPLICATIVE CORRECTION
        # recompute phi with corrected omega
        t0 = time.time()
        plasma.phi = hw.get_phi(plasma.omega, parameters["dx"], x0=plasma.phi)
        total_time_poisson = time.time() - t0
        # print(
        #     "plasma.omega mean:",
        #     math.mean(plasma.omega),
        #     "plasma.omega std:",
        #     math.std(plasma.omega),
        # )
        # print(
        #     "plasma.phi mean:",
        #     math.mean(plasma.phi),
        #     "plasma.phi std:",
        #     math.std(plasma.phi),
        # )

        # print("total_time_native_call", total_time_native_call, "total_time_poisson", total_time_poisson)
        return plasma

    return corrector_step


def native_call_CUSTOM(f: Callable, *inputs: math.Tensor, channels_last, channel_dim='vector', spatial_dim=None):
    """
    Calls `f` with the native representations of the `inputs` tensors in standard layout and returns the result as a `Tensor`.

    All inputs are converted to native tensors (including precision cast) depending on `channels_last`:

    * `channels_last=True`: Dimension layout `(total_batch_size, spatial_dims..., total_channel_size)`
    * `channels_last=False`: Dimension layout `(total_batch_size, total_channel_size, spatial_dims...)`

    All batch dimensions are compressed into a single dimension with `total_batch_size = input.shape.batch.volume`.
    The same is done for all channel dimensions.

    Additionally, missing batch and spatial dimensions are added so that all `inputs` have the same batch and spatial shape.

    Args:
        f: Function to be called on native tensors of `inputs`.
            The function output must have the same dimension layout as the inputs, unless overridden by `spatial_dim`,
            and the batch size must be identical.
        *inputs: Uniform `Tensor` arguments
        channels_last: (Optional) Whether to put channels as the last dimension of the native representation.
            If `None`, the channels are put in the default position associated with the current backend,
            see `phiml.math.backend.Backend.prefers_channels_last()`.
        channel_dim: Name of the channel dimension of the result.
        spatial_dim: Name of the spatial dimension of the result.

    Returns:
        `Tensor` with batch and spatial dimensions of `inputs`, unless overridden by `spatial_dim`,
        and single channel dimension `channel_dim`.
    """
    # if channels_last is None:
    #     try:
    #         backend = math.choose_backend(f)
    #     except NoBackendFound:
    #         # backend = choose_backend_t(*inputs, prefer_default=True) #WHAT IS choose_backend_t??
    #         pass
    #     channels_last = backend.prefers_channels_last()
    batch = math.merge_shapes(*[i.shape.batch for i in inputs])
    spatial = math.merge_shapes(*[i.shape.spatial for i in inputs])
    natives = []
    for i in inputs:
        groups = (batch, *i.shape.spatial.names, i.shape.channel) if channels_last else (batch, i.shape.channel, *i.shape.spatial.names)
        natives.append(math.reshaped_native(i, groups, force_expand=False))
    print("native_call_CUSTOM natives:", natives)
    output = f(*natives)
    if isinstance(channel_dim, str):
        channel_dim = math.channel(channel_dim)
    assert isinstance(channel_dim, math.Shape), "channel_dim must be a Shape or str"
    if isinstance(output, (tuple, list)):
        raise NotImplementedError()
    else:
        if spatial_dim is None:
            groups = (batch, *spatial, channel_dim) if channels_last else (batch, channel_dim, *spatial)
        else:
            if isinstance(spatial_dim, str):
                spatial_dim = math.spatial(spatial_dim)
            assert isinstance(spatial_dim, math.Shape), "spatial_dim must be a Shape or str"
            groups = (batch, *spatial_dim, channel_dim) if channels_last else (batch, channel_dim, *spatial_dim)
        result = math.reshaped_tensor(output, groups, convert=False)
        if result.shape.get_size(channel_dim.name) == 1 and not channel_dim.item_names[0]:
            result = result.dimension(channel_dim.name)[0]  # remove vector dim if not required
        return result
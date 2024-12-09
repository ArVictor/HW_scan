"""
Defines the loss function for the predictor-corrector scheme with machine learning, using PhiML.
=====================================================

This module provides the loss function for the predictor-corrector scheme.

get_loss_function returns the callable loss function.
"""

# Define Frame Properties
# Assume (..., y, x) as shape
from typing import Callable
from phiml import math
from phiml import nn
import tensorflow as tf


def get_loss_function(
    unrolled_steps: int,
    std_omega: float,
    std_density: float,
    std_phi: float,
    integration_step: Callable,
    parameters: dict,
    corrector_step: Callable,
    network: nn.Network,
    temporal_causality_eps: float,
    
) -> Callable:
    """
    Defines the loss function and returns it.

    Args:
        unrolled_steps (int): Number of time integrations to unroll.
        std_omega (float): Standard deviation of omega.
        std_density (float): Standard deviation of density.
        std_phi (float): Standard deviation of phi.
        integration_step (Callable): Callable solver (usually euler_step or rk4_step).
        parameters (dict): Dictionary with the simulation parameters
        corrector_step (Callable): Function returning the corrected plasma from a given plasma, network input.
        network (Network): Neural network for the corrector step.
        temporal_causality_eps (float): Weight of temporal causality for the loss computation. if <=0, temporal causality is not used. See https://arxiv.org/pdf/2203.07404.pdf RESPECTING CAUSALITY IS ALL YOU NEED FOR TRAINING PHYSICS-INFORMED NEURAL NETWORKS.

    Returns:
        Callable: Loss function that returns (loss, plasma)
    """

    def loss_function(plasma_in, label):
        # print(
        #     "LOSS INPUT plasma_in.phi",
        #     math.std(plasma_in.phi),
        # )
        error = 0.0
        # print(
        #     "std_omega:",
        #     std_omega,
        #     "std_density:",
        #     std_density,
        #     "std_phi:",
        #     std_phi,
        # )
        # print("parameters:", parameters)
        # Physical time step
        print("plasma_in", plasma_in)
        print("type(plasma_in)", type(plasma_in))
        #plasma = math.Dict(math.stop_gradient(integration_step(
        #    plasma_in, dt=parameters["dt"], dx=parameters["dx"]
        #)))  # we do the first step outside to have a new variable plasma. Otherwise we overwrite plasma_in in the training data!
        plasma = integration_step(
            plasma_in, dt=parameters["dt"], dx=parameters["dx"]
        )  # we do the first step outside to have a new variable plasma. Otherwise we overwrite plasma_in in the training data!
        print("plasma", plasma)
        print("type(plasma)", type(plasma))
        list_temporal_errors = []
        for step in range(unrolled_steps):  # For each unrolled step
            if step != 0:  # First step was already done.
                # Physical time step
                #plasma = math.Dict(math.stop_gradient(integration_step(
                #    plasma, dt=parameters["dt"], dx=parameters["dx"]
                #)))
                plasma = integration_step(
                    plasma, dt=parameters["dt"], dx=parameters["dx"]
                )
                print("plasma", plasma)

            # print(
            #     "INTEGRATION LOSS plasma.omega",
            #     math.std(plasma.omega),
            #     math.std(label[step][0]),
            #     math.l2_loss((plasma.omega - label[step][0]) / std_omega)
            #     / math.prod(plasma.density.shape.spatial.sizes),
            # )
            # print(
            #     "INTEGRATION LOSS plasma.density",
            #     math.std(plasma.density),
            #     math.std(label[step][1]),
            #     math.l2_loss((plasma.density - label[step][1]) / std_density)
            #     / math.prod(plasma.density.shape.spatial.sizes),
            # )
            # print(
            #     "INTEGRATION LOSS plasma.phi",
            #     math.std(plasma.phi),
            #     math.std(label[step][2]),
            #     math.l2_loss((plasma.phi - label[step][2]) / std_phi)
            #     / math.prod(plasma.density.shape.spatial.sizes),
            # )
            # ML corrector step
            plasma = corrector_step(plasma, network)
            # print(
            #     "corrector plasma_state.phi",
            #     math.std(plasma.phi),
            #     math.std(label[step][2]),
            # )
            # print(
            #     "LOSS plasma.omega",
            #     math.std(plasma.omega),
            #     math.std(label[step][0]),
            #     math.l2_loss((plasma.omega - label[step][0]) / std_omega)
            #     / math.prod(plasma.density.shape.spatial.sizes),
            # )
            # print(
            #     "LOSS plasma.density",
            #     math.std(plasma.density),
            #     math.std(label[step][1]),
            #     math.l2_loss((plasma.density - label[step][1]) / std_density)
            #     / math.prod(plasma.density.shape.spatial.sizes),
            # )
            # print(
            #     "LOSS plasma.phi",
            #     math.std(plasma.phi),
            #     math.std(label[step][2]),
            #     math.l2_loss((plasma.phi - label[step][2]) / std_phi)
            #     / math.prod(plasma.density.shape.spatial.sizes),
            # )
            # Add fields' L2-loss (normalized w.r.t. their individual standard deviations) to the total loss
            print("loss_step:",
                math.l2_loss((plasma.omega - label[step][0]) / std_omega)
                + math.l2_loss((plasma.density - label[step][1]) / std_density)
                + math.l2_loss((plasma.phi - label[step][2]) / std_phi))
            list_temporal_errors.append((
                math.l2_loss((plasma.omega - label[step][0]) / std_omega)
                + math.l2_loss((plasma.density - label[step][1]) / std_density)
                + math.l2_loss((plasma.phi - label[step][2]) / std_phi)
            ))
            # error = error + (
            #     math.l2_loss((plasma.omega - label[step][0]) / std_omega)
            #     + math.l2_loss((plasma.density - label[step][1]) / std_density)
            #     + math.l2_loss((plasma.phi - label[step][2]) / std_phi)
            # )
        for step in range(unrolled_steps):
            if temporal_causality_eps>0.0:
                sum_previous_errors = 0.0
                for i in range(step):
                    sum_previous_errors = sum_previous_errors + list_temporal_errors[i]
                weight_temporal_causality = math.exp(-temporal_causality_eps * sum_previous_errors)
                print("weight_temporal_causality:", weight_temporal_causality)
                error = error + weight_temporal_causality*list_temporal_errors[step]
            else:
                error = error + list_temporal_errors[step]
            
        error = error / 3.0  # Normalize w.r.t. the number of fields
        error = error / unrolled_steps  # Normalize w.r.t. the number of unrolled steps
        error = error / math.prod(
            plasma.density.shape.spatial.sizes
        )  # Normalize w.r.t. the fields sizes
        # update_weights does the sum over the batch dimension. (actually over all dimensions still present in the loss).
        error = math.mean(error, dim=math.batch)  # We do the average over the batches
        # Robin uses reduce_sum for the model loss
        if hasattr(network, 'losses'):
            error = error + tf.reduce_mean(
                network.losses
            )  # mean instead of sum so that a longer network has the same loss scale than one with fewer layers
        else:
            error = error + tf.reduce_mean(
                network.__wrapped__.losses
            )  # mean instead of sum so that a longer network has the same loss scale than one with fewer layers
        return error, plasma

    return loss_function

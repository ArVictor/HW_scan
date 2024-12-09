"""
This module defines a function for building a ResNet-like convolutional neural network using TensorFlow.
The provided function, build_resnet, constructs a ResNet model with customizable parameters such as input and output channels,
number of layers, kernel size, use of batch normalization, and choice of activation function.

The ResNet architecture is constructed with convolutional blocks, and the module also includes utility functions
for handling periodic padding in 2D tensors.

Example usage:
    # Define Frame Properties
    # Assume (..., y, x) as shape
    from resnet_builder import build_resnet

    in_channels = 1 # Just phi
    out_channels = 2    # density and omega
    layers = [32, 32, 32]
    resnet_model = build_resnet(in_channels, out_channels, layers)
"""

# Define Frame Properties
# Assume (..., y, x) as shape
from typing import Sequence, Callable
import tensorflow as tf

from tf_hw2d.neural_network import tf_tools


def build_resnet(
    in_channels: int,
    out_channels: int,
    layers: Sequence[int],
    kernel_size: int = 5,
    batch_norm: bool = False,
    activation_layer: Callable = tf.keras.layers.LeakyReLU,
):
    params = dict(
        kernel_size=kernel_size,
        padding="valid",
        kernel_initializer="glorot_uniform",
        kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00001, l2=0.00001),
        bias_regularizer=tf.keras.regularizers.L1L2(l1=0.00001, l2=0.00001),
    )
    l_input = tf.keras.layers.Input(
        shape=(
            None,
            None,
            in_channels,
        )
    )

    block_list = []
    x = l_input
    x = tf_tools.periodic_padding_2D(x, padding=int((kernel_size - 1) / 2))
    block_0 = tf.keras.layers.Conv2D(filters=layers[0], use_bias=True, **params)(x)
    block_0 = activation_layer()(block_0)
    # if batch_norm: #TEST PRERESNET
    #    block_0 = tf.keras.layers.BatchNormalization()(block_0)
    block_list.append(block_0)

    for i in range(1, len(layers)):
        # Wrap padding in x and y
        x = block_list[-1]  # Take previous output
        if batch_norm: #TEST PRERESNET
            # x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LayerNormalization()(x)
        x = activation_layer()(x) #TEST PRERESNET
        x = tf_tools.periodic_padding_2D(x, padding=int((kernel_size - 1) / 2))
        # First Conv2D & LeakyReLU
        l_conv1 = tf.keras.layers.Conv2D(
           filters=layers[i],
           use_bias=True,
           **params,
        )(x)
        # l_conv1 = tf.keras.layers.SeparableConv2D(
        #     filters=layers[i],
        #     use_bias=True,
        #     **params,
        # )(x)
        # l_conv1 = activation_layer()(l_conv1) #TEST PRERESNET
        # if batch_norm: #TEST PRERESNET
        #     l_conv1 = tf.keras.layers.BatchNormalization()(l_conv1)
        # Wrap padding for second layer
        x = l_conv1
        if batch_norm: #TEST PRERESNET
            # x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.LayerNormalization()(x)
        x = activation_layer()(x) #TEST PRERESNET
        x = tf.keras.layers.Dropout(.2)(x) #TEST DROPOUT
        x = tf_tools.periodic_padding_2D(x, padding=int((kernel_size - 1) / 2))
        l_conv2 = tf.keras.layers.Conv2D(
            filters=layers[i],
            use_bias=True,
            **params,
        )(
            x
        )  # NOTE: no relu here
        # l_conv2 = tf.keras.layers.SeparableConv2D(
        #     filters=layers[i],
        #     use_bias=True,
        #     **params,
        # )(x)  # NOTE: no relu here
        # Add and add Activation
        l_skip1 = tf.keras.layers.add([block_list[-1], l_conv2])
        # l_skip1 = tf.keras.layers.add([*block_list, l_conv2])   #DENSENET
        # block_1 = activation_layer()(l_skip1) #TEST PRERESNET
        block_1 = l_skip1 #TEST PRERESNET
        block_list.append(block_1)  # keep track of blocks

    # Wrap up
    x = block_list[-1]  # Take previous output
    # Last convolution for output
    x = tf_tools.periodic_padding_2D(x, padding=int((kernel_size - 1) / 2))
    l_output = tf.keras.layers.Conv2D(
        filters=out_channels,
        use_bias=True,
        **params,
    )(x)

    return tf.keras.models.Model(inputs=l_input, outputs=l_output)

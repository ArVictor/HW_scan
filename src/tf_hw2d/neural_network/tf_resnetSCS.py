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


def build_resnetSCS(
    in_channels: int,
    out_channels: int,
    layers: Sequence[int],
    kernel_size: int = 5,
    batch_norm: bool = False,
    activation_layer: Callable = tf.keras.layers.LeakyReLU,
    input_x_z: int = 64,
):
    params = dict(
        kernel_size=kernel_size,
        padding="valid",
        # kernel_initializer="glorot_uniform",
        # kernel_regularizer=tf.keras.regularizers.L1L2(l1=0.00001, l2=0.00001),
        #bias_regularizer=tf.keras.regularizers.L1L2(l1=0.00001, l2=0.00001),
    )
    l_input = tf.keras.layers.Input(
        shape=(
            input_x_z,
            input_x_z,
            in_channels,
        )
    )

    block_list = []
    x = l_input
    x = tf_tools.periodic_padding_2D(x, padding=int((kernel_size - 1) / 2))
    #block_0 = tf.keras.layers.Conv2D(filters=layers[0], use_bias=False, **params)(x)
    block_0 = CosSim2D(units=layers[0], **params)(x)
    # block_0 = activation_layer()(block_0)
    # if batch_norm: #TEST PRERESNET
    #    block_0 = tf.keras.layers.BatchNormalization()(block_0)
    block_list.append(block_0)

    for i in range(1, len(layers)):
        # Wrap padding in x and y
        x = block_list[-1]  # Take previous output
        if batch_norm: #TEST PRERESNET
            x = tf.keras.layers.BatchNormalization()(x)
        # x = activation_layer()(x) #TEST PRERESNET
        x = tf_tools.periodic_padding_2D(x, padding=int((kernel_size - 1) / 2))
        # First Conv2D & LeakyReLU
        # l_conv1 = tf.keras.layers.Conv2D(
        #    filters=layers[i],
        #    use_bias=False,
        #    **params,
        # )(x)
        l_conv1 = CosSim2D(units=layers[i], **params)(x)
        # l_conv1 = tf.keras.layers.SeparableConv2D(
        #     filters=layers[i],
        #     use_bias=False,
        #     **params,
        # )(x)
        # l_conv1 = activation_layer()(l_conv1) #TEST PRERESNET
        # if batch_norm: #TEST PRERESNET
        #     l_conv1 = tf.keras.layers.BatchNormalization()(l_conv1)
        # Wrap padding for second layer
        x = l_conv1
        if batch_norm: #TEST PRERESNET
            x = tf.keras.layers.BatchNormalization()(x)
        # x = activation_layer()(x) #TEST PRERESNET
        x = tf.keras.layers.Dropout(.2)(x) #TEST DROPOUT
        x = tf_tools.periodic_padding_2D(x, padding=int((kernel_size - 1) / 2))
        # l_conv2 = tf.keras.layers.Conv2D(
        #     filters=layers[i],
        #     use_bias=False,
        #     **params,
        # )(
        #     x
        # )  # NOTE: no relu here
        l_conv2 = CosSim2D(units=layers[i], **params)(x)
        # l_conv2 = tf.keras.layers.SeparableConv2D(
        #     filters=layers[i],
        #     use_bias=False,
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
    # l_output = tf.keras.layers.Conv2D(
    #     filters=out_channels,
    #     use_bias=True,
    #     **params,
    # )(x)
    l_output = CosSim2D(units=out_channels, **params)(x)    #Let's try a normal conv to finishso that it learn the bias

    return tf.keras.models.Model(inputs=l_input, outputs=l_output)



import math as python_math  #to not mix it with phiml.math

import tensorflow as tf


class CosSim2D(tf.keras.layers.Layer):
    def __init__(self, kernel_size, units=32, stride=1, depthwise_separable=False, padding='valid'):
        super(CosSim2D, self).__init__()
        self.depthwise_separable = depthwise_separable
        self.units = units
        assert kernel_size in [1, 3, 5], "kernel of this size not supported"
        self.kernel_size = kernel_size
        if self.kernel_size == 1:
            self.stack = lambda x: x
        elif self.kernel_size == 3:
            self.stack = self.stack3x3
        elif self.kernel_size == 5:
            self.stack = self.stack5x5
        self.stride = stride
        if padding == 'same':
            self.pad = self.kernel_size // 2
            self.pad_1 = 1
            self.clip = 0
        elif padding == 'valid':
            self.pad = 0
            self.pad_1 = 0
            self.clip = self.kernel_size // 2

    def build(self, input_shape):
        self.in_shape = input_shape
        self.out_y = python_math.ceil((self.in_shape[1] - 2 * self.clip) / self.stride)
        self.out_x = python_math.ceil((self.in_shape[2] - 2 * self.clip) / self.stride)
        self.flat_size = self.out_x * self.out_y
        self.channels = self.in_shape[3]

        if self.depthwise_separable:
            self.w = self.add_weight(
                shape=(1, tf.square(self.kernel_size), self.units),
                initializer="glorot_uniform", name='w',
                trainable=True,
                regularizer=tf.keras.regularizers.L1L2(l1=0.00001, l2=0.00001),
            )
        else:
            self.w = self.add_weight(
                shape=(1, self.channels * tf.square(self.kernel_size), self.units),
                initializer="glorot_uniform", name='w',
                trainable=True,
                regularizer=tf.keras.regularizers.L1L2(l1=0.00001, l2=0.00001),
            )

        p_init = tf.constant_initializer(value=100**0.5)
        self.p = self.add_weight(
            shape=(self.units,), initializer=p_init, trainable=True, name='p')

        q_init = tf.constant_initializer(value=10**0.5)
        self.q = self.add_weight(
            shape=(1,), initializer=q_init, trainable=True, name='q')

    def l2_normal(self, x, axis=None, epsilon=1e-12):
        square_sum = tf.reduce_sum(tf.square(x), axis, keepdims=True)
        x_inv_norm = tf.sqrt(tf.maximum(square_sum, epsilon))
        return x_inv_norm

    def stack3x3(self, image):
        '''
            sliding window implementation for 3x3 kernel
        '''
        x = tf.shape(image)[2]
        y = tf.shape(image)[1]
        stack = tf.stack(
            [
                tf.pad(  # top row
                    image[:, :y - 1 - self.clip:, :x - 1 - self.clip, :],
                    tf.constant([[0, 0], [self.pad, 0], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, :y - 1 - self.clip, self.clip:x - self.clip, :],
                    tf.constant([[0, 0], [self.pad, 0], [0, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, :y - 1 - self.clip, 1 + self.clip:, :],
                    tf.constant([[0, 0], [self.pad, 0], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],

                tf.pad(  # middle row
                    image[:, self.clip:y - self.clip, :x - 1 - self.clip, :],
                    tf.constant([[0, 0], [0, 0], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                image[:, self.clip:y - self.clip:self.stride, self.clip:x - self.clip:self.stride, :],
                tf.pad(
                    image[:, self.clip:y - self.clip, 1 + self.clip:, :],
                    tf.constant([[0, 0], [0, 0], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],

                tf.pad(  # bottom row
                    image[:, 1 + self.clip:, :x - 1 - self.clip, :],
                    tf.constant([[0, 0], [0, self.pad], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1 + self.clip:, self.clip:x - self.clip, :],
                    tf.constant([[0, 0], [0, self.pad], [0, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1 + self.clip:, 1 + self.clip:, :],
                    tf.constant([[0, 0], [0, self.pad], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :]
            ], axis=3)
        return stack

    def stack5x5(self, image):
        '''
            sliding window implementation for 5x5 kernel
        '''
        x = tf.shape(image)[2]
        y = tf.shape(image)[1]
        stack = tf.stack(
            [
                tf.pad(  # top row
                    image[:, :y - 2 - self.clip:, :x - 2 - self.clip, :],
                    tf.constant([[0, 0], [self.pad, 0], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, :y - 2 - self.clip:, 1:x - 1 - self.clip, :],
                    tf.constant([[0, 0], [self.pad, 0], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, :y - 2 - self.clip:, self.clip:x - self.clip, :],
                    tf.constant([[0, 0], [self.pad, 0], [0, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, :y - 2 - self.clip:, 1 + self.clip:-1, :],
                    tf.constant([[0, 0], [self.pad, 0], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, :y - 2 - self.clip:, 2 + self.clip:, :],
                    tf.constant([[0, 0], [self.pad, 0], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],

                tf.pad(  # 2nd row
                    image[:, 1:y - 1 - self.clip:, :x - 2 - self.clip, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1:y - 1 - self.clip:, 1:x - 1 - self.clip, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1:y - 1 - self.clip:, self.clip:x - self.clip, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [0, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1:y - 1 - self.clip:, 1 + self.clip:-1, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1:y - 1 - self.clip:, 2 + self.clip:, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],

                tf.pad(  # 3rd row
                    image[:, self.clip:y - self.clip, :x - 2 - self.clip, :],
                    tf.constant([[0, 0], [0, 0], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, self.clip:y - self.clip, 1:x - 1 - self.clip, :],
                    tf.constant([[0, 0], [0, 0], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                image[:, self.clip:y - self.clip, self.clip:x - self.clip, :][:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, self.clip:y - self.clip, 1 + self.clip:-1, :],
                    tf.constant([[0, 0], [0, 0], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, self.clip:y - self.clip, 2 + self.clip:, :],
                    tf.constant([[0, 0], [0, 0], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],

                tf.pad(  # 4th row
                    image[:, 1 + self.clip:-1, :x - 2 - self.clip, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1 + self.clip:-1, 1:x - 1 - self.clip, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1 + self.clip:-1, self.clip:x - self.clip, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [0, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1 + self.clip:-1, 1 + self.clip:-1, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 1 + self.clip:-1, 2 + self.clip:, :],
                    tf.constant([[0, 0], [self.pad_1, self.pad_1], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],

                tf.pad(  # 5th row
                    image[:, 2 + self.clip:, :x - 2 - self.clip, :],
                    tf.constant([[0, 0], [0, self.pad], [self.pad, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 2 + self.clip:, 1:x - 1 - self.clip, :],
                    tf.constant([[0, 0], [0, self.pad], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 2 + self.clip:, self.clip:x - self.clip, :],
                    tf.constant([[0, 0], [0, self.pad], [0, 0], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 2 + self.clip:, 1 + self.clip:-1, :],
                    tf.constant([[0, 0], [0, self.pad], [self.pad_1, self.pad_1], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
                tf.pad(
                    image[:, 2 + self.clip:, 2 + self.clip:, :],
                    tf.constant([[0, 0], [0, self.pad], [0, self.pad], [0, 0]])
                )[:, ::self.stride, ::self.stride, :],
            ], axis=3)
        return stack

    def call_body(self, inputs):
        channels = tf.shape(inputs)[-1]
        x = self.stack(inputs)
        x = tf.reshape(x, (-1, self.flat_size, channels * tf.square(self.kernel_size)))
        x_norm = (self.l2_normal(x, axis=2) + tf.square(self.q)/10)
        w_norm = (self.l2_normal(self.w, axis=1) + tf.square(self.q)/10)
        x = tf.matmul(x / x_norm, self.w / w_norm)
        sign = tf.sign(x)
        x = tf.abs(x) + 1e-12
        x = tf.pow(x, tf.square(self.p)/100)
        x = sign * x
        x = tf.reshape(x, (-1, self.out_y, self.out_x, self.units))
        return x

    @tf.function
    def call(self, inputs, training=None):
        if self.depthwise_separable:
            x = tf.vectorized_map(self.call_body, tf.expand_dims(tf.transpose(inputs, (3, 0, 1, 2)), axis=-1),
                                  fallback_to_while_loop=True)
            x = tf.transpose(x, (1, 2, 3, 4, 0))
            x = tf.reshape(x, (-1, self.out_y, self.out_x, self.channels * self.units))
            return x
        else:
            x = self.call_body(inputs)
            return x
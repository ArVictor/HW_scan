"""
This module provides utility functions for handling tensors with TensorFlow.

Functions:
    - get_valid(tensor, cut_size): Extracts the central region of a tensor by removing a specified number of cells from each side.
    - periodic_padding_flexible(tensor, axis, padding=1): Adds periodic padding to a tensor along the specified axis or axes.
    - periodic_padding_2D(tensor, padding=1): Adds 2D periodic padding to a tensor by applying periodic padding along both spatial axes.

These utility functions are useful for tasks involving tensor manipulation, especially in the context of deep learning with TensorFlow.

"""

import tensorflow as tf


def get_valid(tensor, cut_size):
    return tensor[:, cut_size:-cut_size, cut_size:-cut_size, :]


def periodic_padding_flexible(tensor, axis, padding=1):
    """
        add periodic padding to a tensor for specified axis
        tensor: input tensor
        axis: on or multiple axis to pad along, int or tuple
        padding: number of cells to pad, int or tuple

        return: padded tensor
    https://stackoverflow.com/questions/39088489/tensorflow-periodic-padding
    """

    if isinstance(axis, int):
        axis = (axis,)
    if isinstance(padding, int):
        padding = (padding,)

    ndim = len(tensor.shape)
    for ax, p in zip(axis, padding):
        # create a slice object that selects everything from all axes,
        # except only 0:p for the specified for right, and -p: for left
        ind_right = [slice(-p, None) if i == ax else slice(None) for i in range(ndim)]
        ind_left = [slice(0, p) if i == ax else slice(None) for i in range(ndim)]
        right = tensor[ind_right]
        left = tensor[ind_left]
        middle = tensor
        tensor = tf.concat([right, middle, left], axis=ax)

    return tensor


def periodic_padding_2D(tensor, padding=1):
    """
    add 2D periodic padding to a tensor
    tensor: input tensor
    padding: number of cells to pad, int or tuple

    return: 2D padded tensor
    """

    x = periodic_padding_flexible(tensor, 1, padding=padding)
    x = periodic_padding_flexible(x, 2, padding=padding)
    return x


def weight_averaging(current_cnn, saved_average, n_model_average):
    """
    Computes the average of the current model and a saved running average model.
    current_cnn: Model
    saved_average: File name of the saved average model.
    n_model_average: Number of models in the average.

    return: New averaged model
    """
    w = current_cnn.get_weights()
    average_cnn = tf.keras.models.load_model(saved_average)
    average_w = average_cnn.get_weights()
    updated_weights = [
        (element_w + n_model_average * element_average_w) / (n_model_average + 1)
        for element_w, element_average_w in zip(w, average_w)
    ]
    current_cnn.set_weights(updated_weights)

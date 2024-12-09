"""
TensorFlow implementation of the unified machine learning API.
Equivalent functions also exist for the other frameworks.

For API documentation, see `phiml.nn`.
"""
from typing import Sequence, Callable
import tensorflow as tf

# ORIGINAL FUNCTION FROM PHIML.MATH
# ORIGINAL FUNCTION FROM PHIML.MATH
# def update_weights(net: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, loss_function: Callable, *loss_args, **loss_kwargs):
#     with tf.GradientTape() as tape:
#         output = loss_function(*loss_args, **loss_kwargs)
#         loss = output[0] if isinstance(output, tuple) else output
#         gradients = tape.gradient(loss.sum, net.trainable_variables)
#     optimizer.apply_gradients(zip(gradients, net.trainable_variables))
#     return output



def update_weights_SAM(net: tf.keras.Model, optimizer: tf.keras.optimizers.Optimizer, loss_function: Callable, rho: float, *loss_args, **loss_kwargs):
    e_ws = []   #gradients list to substract them later
    #get current gradient
    with tf.GradientTape() as tape:
        output = loss_function(*loss_args, **loss_kwargs)
        loss = output[0] if isinstance(output, tuple) else output
        gradients = tape.gradient(loss.sum, net.trainable_variables)
    grad_norm = tf.linalg.global_norm(gradients) # Get gradient norm. The gradient ascent step is done with fixed norm rho
    #Gradient ASCENT step
    for i in range(len(net.trainable_variables)):
        e_w = gradients[i] * rho / (grad_norm + 1e-12)
        net.trainable_variables[i].assign_add(e_w)
        e_ws.append(e_w)
    #Get SAM gradient
    with tf.GradientTape() as tape:
        output = loss_function(*loss_args, **loss_kwargs)
        loss = output[0] if isinstance(output, tuple) else output # + args.weight_decay*tf.add_n([tf.nn.l2_loss(v) for v in model.trainable_variables if '_bn' not in v.name])??? where is it?
        sam_gradients = tape.gradient(loss.sum, net.trainable_variables)
    #Undo gradient scent step
    for i in range(len(net.trainable_variables)):
        net.trainable_variables[i].assign_add(-e_ws[i])
    
    optimizer.apply_gradients(zip(sam_gradients, net.trainable_variables))
    return output




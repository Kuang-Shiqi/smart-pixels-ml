import keras
from keras.layers import *
from keras.models import Sequential, Model
from keras.utils import Sequence
from qkeras import *

import tensorflow as tf
from tensorflow.keras import datasets, layers, models

def var_network(var, hidden=10, output=2):
    var = Flatten(name="flatten")(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="dense_1"
    )(var)
    #var = keras.activations.tanh(var)
    var = QActivation("quantized_tanh(8, 0, 1)", name="activation_tanh_2")(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="dense_2"
    )(var)
    #var = keras.activations.tanh(var)
    var = QActivation("quantized_tanh(8, 0, 1)", name="activation_tanh_3")(var)
    return QDense(
        output,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        name="dense_3"
    )(var)

def conv_network(var, kernel_size=3):
    # Get actual dimensions from the input tensor
    nrows = var.shape[1]  # Height dimension
    ncols = var.shape[2]  # Width dimension  
    timeslices = var.shape[3]  # Channel dimension
    
    # Project along width (collapse width dimension)
    proj_x = AveragePooling2D(
        pool_size=(1, ncols),  # Use actual width instead of hardcoded 21
        strides=None,
        padding="valid",
        data_format=None,
        name="avg_pooling_2d_proj_x"
    )(var)
    proj_x = Reshape((nrows, timeslices), name="reshape_proj_x")(proj_x)
    
    # Project along height (collapse height dimension)
    proj_y = AveragePooling2D(
        pool_size=(nrows, 1),  # Use actual height instead of hardcoded 13
        strides=None,
        padding="valid",
        data_format=None,
        name="avg_pooling_2d_proj_y"
    )(var)
    proj_y = Reshape((ncols, timeslices), name="reshape_proj_y")(proj_y)

    proj_x = QConv1D(
        5, kernel_size,
        kernel_quantizer=quantized_bits(4, 0, 1, alpha=1),
        bias_quantizer=quantized_bits(4, 0, 1, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        bias_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="conv1d_proj_x"
    )(proj_x)

    proj_y = QConv1D(
        5, kernel_size,
        kernel_quantizer=quantized_bits(4, 0, 1, alpha=1),
        bias_quantizer=quantized_bits(4, 0, 1, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        bias_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="conv1d_proj_y"
    )(proj_y)

    var = Concatenate(axis=1, name="concatenate")([proj_x, proj_y])

    #var = keras.activations.tanh(var)
    var = QActivation("quantized_tanh(4, 0, 1)", name="activation_tanh_1")(var)

    return var

def CreateModel(shape):
    x_base = x_in = Input(shape, name="input_pxls/")
    stack = conv_network(x_base)
    #stack = AveragePooling2D(
    #    pool_size=(2, 2),
    #    strides=None,
    #    padding="valid",
    #    data_format=None,
    #)(stack)
    #stack = QActivation("quantized_bits(8, 0, alpha=1)")(stack)
    stack = var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model
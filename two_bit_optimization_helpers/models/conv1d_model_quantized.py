import tensorflow as tf
from keras.layers import (
    Input, Flatten, AveragePooling2D, Reshape,
    Concatenate, Activation
)
from keras.models import Model
from qkeras import (
    QDense, QConv1D, QActivation,
    quantized_bits
)
from SoftQuantizeLayer import SoftQuantizeLayer

def _var_network(var, hidden=10, output=2):
    var = Flatten(name="flatten")(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="dense_1"
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)", name="activation_tanh_2")(var)
    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="dense_2"
    )(var)
    var = QActivation("quantized_tanh(8, 0, 1)", name="activation_tanh_3")(var)
    return QDense(
        output,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        name="dense_3"
    )(var)

def _conv_network(var, kernel_size=3):
    nrows = var.shape[1] # 13, for now
    ncols = var.shape[2] # 20, for now
    timeslices = var.shape[3] # either 20 or 2, for now
    proj_x = AveragePooling2D(
        pool_size=(1, 16),
        strides=None,
        padding="valid",
        data_format=None,
        name="avg_pooling_2d_proj_x"
    )(var)
    proj_x = Reshape((nrows, timeslices), name="reshape_proj_x")(proj_x)
    proj_y = AveragePooling2D(
        pool_size=(nrows, 1),
        strides=None,
        padding="valid",
        data_format=None,
        name="avg_pooling_2d_proj_y"
    )(var)
    proj_y = Reshape((ncols, timeslices), name="reshape_proj_y")(proj_y)

    proj_x = QConv1D(
        5,kernel_size,
        kernel_quantizer=quantized_bits(4, 0, 1, alpha=1),
        bias_quantizer=quantized_bits(4, 0, 1, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        bias_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="conv1d_proj_x"
    )(proj_x)

    proj_y = QConv1D(
        5,kernel_size,
        kernel_quantizer=quantized_bits(4, 0, 1, alpha=1),
        bias_quantizer=quantized_bits(4, 0, 1, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        bias_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="conv1d_proj_y"
    )(proj_y)

    var = Concatenate(axis=1, name="concatenate")([proj_x, proj_y])
    var = QActivation("quantized_tanh(4, 0, 1)", name="activation_tanh_1")(var)

    return var

def QConv1D_Full(shape):
    x_base = x_in = Input(shape, name="input_pxls")
    stack = _conv_network(x_base)
    stack = _var_network(stack, hidden=16, output=8)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model

def QConv1D_Slim(shape):
    x_base = x_in = Input(shape, name="input_pxls")
    stack = _conv_network(x_base)
    stack = _var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model

def QConv1D_Full_SoftQuantizer(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    x_base = x_in = Input(shape, name="input_pxls")
    x_base = SoftQuantizeLayer(
        n_bits=2,                     
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,                
        trainable_k=True,             
        name='soft_quantizer_output'  
    )(x_base)
    stack = _conv_network(x_base)
    stack = _var_network(stack, hidden=16, output=8)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model

def QConv1D_Slim_SoftQuantizer(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    x_base = x_in = Input(shape, name="input_pxls")
    x_base = SoftQuantizeLayer(
        n_bits=2,                     
        initial_thresholds=initial_thresholds,
        threshold_offset=threshold_offset,
        initial_levels=initial_levels,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,                
        trainable_k=True,             
        name='soft_quantizer_output'  
    )(x_base)
    stack = _conv_network(x_base)
    stack = _var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model

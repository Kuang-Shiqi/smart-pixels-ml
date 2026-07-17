import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Flatten, Dense, Activation,
    Conv2D, SeparableConv2D,
    AveragePooling2D
)
from tensorflow.keras.models import Model
from SoftQuantizeLayer import SoftQuantizeLayer

def _var_network(var, hidden=10, output=2):
    var = Flatten()(var)

    # First Dense layer
    var = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh")(var)

    # Second Dense layer
    var = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh")(var)

    # Output layer
    return Dense(
        output,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    )(var)

def _conv_network(var, n_filters=5, kernel_size=3):
    var = SeparableConv2D(
        n_filters, kernel_size,
        depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh")(var)

    var = Conv2D(
        n_filters, 1,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh")(var)
    return var

def Conv2D_Max(shape, n_filters, pool_size):
    x_base = x_in = Input(shape)
    stack = _conv_network(x_base)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size),
        strides=None,
        padding="valid",
        data_format=None,
    )(stack)
    stack = _var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack)
    return model

def Conv2D_Full(shape, n_filters, pool_size):
    x_base = x_in = Input(shape)
    stack = _conv_network(x_base)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size),
        strides=None,
        padding="valid",
        data_format=None,
    )(stack)
    stack = _var_network(stack, hidden=16, output=8)
    model = Model(inputs=x_in, outputs=stack)
    return model

def Conv2D_Slim(shape, n_filters, pool_size):
    x_base = x_in = Input(shape)
    stack = _conv_network(x_base)
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size),
        strides=None,
        padding="valid",
        data_format=None,
    )(stack)
    stack = _var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack)
    return model

def Conv2D_Max_SoftQuantizer(shape, n_filters, pool_size, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    x_base = x_in = Input(shape)
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
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = _var_network(stack, hidden=16, output=14)
    model = Model(inputs=x_in, outputs=stack)
    return model
    
def Conv2D_Full_SoftQuantizer(shape, n_filters, pool_size, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    x_base = x_in = Input(shape)
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
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = _var_network(stack, hidden=16, output=8)
    model = Model(inputs=x_in, outputs=stack)
    return model

def Conv2D_Slim_SoftQuantizer(shape, n_filters, pool_size, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    x_base = x_in = Input(shape)
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
    stack = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(stack)
    stack = _var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack)
    return model
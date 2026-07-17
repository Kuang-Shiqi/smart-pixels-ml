import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Activation, AveragePooling2D, Flatten,
    Input, Concatenate
)
from tensorflow.keras.models import Model
from SoftQuantizeLayer import SoftQuantizeLayer

def _var_network(var, hidden=10, output=2):
    var = Flatten(name="flatten_var")(var)
    var = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh", name="activation_tanh_1")(var)
    var = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)
    var = Activation("tanh", name="activation_tanh_2")(var)
    return Dense(
        output,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
    )(var)

def _mlp_encoder_network(var, hidden=16, hidden_dimx=16, hidden_dimy=16):
    proj_x = AveragePooling2D(
        pool_size=(1, hidden_dimx), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    proj_x = Flatten()(proj_x)

    proj_y = AveragePooling2D(
        pool_size=(hidden_dimy, 1), 
        strides=None, 
        padding="valid", 
        data_format=None,        
    )(var)
    proj_y = Flatten()(proj_y)

    proj_x = Dense(
        hidden_dimx,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_x)
    proj_x = Activation("relu")(proj_x)

    proj_y = Dense(
        hidden_dimy,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_y)
    proj_y = Activation("relu")(proj_y)

    var = Concatenate(axis=1)([proj_x, proj_y])

    var = Dense(
        hidden,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)

    var = Activation("tanh")(var)
    return var

def Mlp_Full(shape):
    x_base = x_in = Input(shape, name="input_pxls")
    stack = _mlp_encoder_network(x_base)
    stack = _var_network(stack, hidden=16, output=8) 
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model

def Mlp_Slim(shape):
    x_base = x_in = Input(shape, name="input_pxls")
    stack = _mlp_encoder_network(x_base)
    stack = _var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model
    
def Mlp_Full_SoftQuantizer(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    x_base = x_in = Input(shape, name="input_pxls")
    x_base = SoftQuantizeLayer(
        n_bits=2,                     
        initial_thresholds=initial_thresholds,
        initial_levels=initial_levels,
        threshold_offset=threshold_offset,    
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds,
        initial_k=1.0,                
        trainable_k=True,             
        name='soft_quantizer_output'  
    )(x_base)

    stack = _mlp_encoder_network(x_base)
    stack = _var_network(stack, hidden=16, output=8)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model

def Mlp_Slim_SoftQuantizer(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
    x_base = x_in = Input(shape, name="input_pxls")
    x_base = SoftQuantizeLayer(
        n_bits=2,                     
        initial_thresholds=initial_thresholds,
        initial_levels=initial_levels,
        threshold_offset=threshold_offset,
        trainable_levels=False,
        trainable_thresholds=trainable_thresholds, 
        initial_k=1.0,                
        trainable_k=True,             
        name='soft_quantizer_output'  
    )(x_base)

    stack = _mlp_encoder_network(x_base)
    stack = _var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model
import tensorflow as tf
from keras.layers import (
    Input, Flatten, AveragePooling2D,
    Reshape, Concatenate
)
from keras.models import Model
from qkeras import (
    QDense, QConv1D, QActivation, quantized_bits
)
from SoftQuantizeLayer import SoftQuantizeLayer

def _var_network(var, hidden=10, output=2):
    var = Flatten(name="flatten_var")(var)
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

    proj_x = QDense(
        hidden_dimx,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_x)
    proj_x = QActivation("quantized_relu(bits=13, integer=5)")(proj_x)

    proj_y = QDense(
        hidden_dimy,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(proj_y)
    proj_y = QActivation("quantized_relu(bits=13, integer=5)")(proj_y)

    var = Concatenate(axis=1)([proj_x, proj_y])

    var = QDense(
        hidden,
        kernel_quantizer=quantized_bits(8, 0, alpha=1),
        bias_quantizer=quantized_bits(8, 0, alpha=1),
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
    )(var)

    var = QActivation("quantized_tanh(8, 0, 1)")(var)
    return var

def QMlp_Full(shape):
    x_base = x_in = Input(shape, name="input_pxls")
    stack = _mlp_encoder_network(x_base)
    stack = _var_network(stack, hidden=16, output=8)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model
    
def QMlp_Slim(shape):
    x_base = x_in = Input(shape, name="input_pxls")
    stack = _mlp_encoder_network(x_base)
    stack = _var_network(stack, hidden=16, output=3)
    model = Model(inputs=x_in, outputs=stack, name="smrtpxl_regression")
    return model

def QMlp_Full_SoftQuantizer(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
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
    
def QMlp_Slim_SoftQuantizer(shape, initial_thresholds, threshold_offset, initial_levels=None, trainable_thresholds=True):
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
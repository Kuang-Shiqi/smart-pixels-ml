# Test model with delayed quantization - quantize AFTER average pooling
# This preserves spatial information during the critical pooling step

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from models.SoftQuantizeLayer import SoftQuantizeLayer

def CreateSlim1DModel_DelayedQuantization(
    input_shape=(16, 16, 2),
    conv_filters=5,
    conv_kernel_size=3,
    dense1_units=16,
    dense2_units=16,
    output_units=3,
    
    # Delayed quantization parameters
    use_delayed_quantization=True,
    soft_quant_n_bits=2,
    soft_quant_initial_range=[-1.0, 1.0],
    soft_quant_trainable_levels=False,
    soft_quant_trainable_bins=True,
    soft_quant_initial_k=1.0,
    soft_quant_trainable_k=True,
):
    """
    Slim1D model with quantization AFTER average pooling.
    This preserves spatial correlations needed for β prediction.
    """
    
    # Input layer - NO quantization here
    input_layer = Input(shape=input_shape, name="input_pxls/")
    
    # FIRST: Do average pooling on FULL PRECISION data
    # This preserves spatial information for β prediction
    avg_pooling_2d_proj_x = AveragePooling2D(
        pool_size=(1, input_shape[1]), 
        name="avg_pooling_2d_proj_x"
    )(input_layer)  # Use original input, not quantized
    
    avg_pooling_2d_proj_y = AveragePooling2D(
        pool_size=(input_shape[0], 1), 
        name="avg_pooling_2d_proj_y"
    )(input_layer)  # Use original input, not quantized
    
    # Reshape to 1D projections
    reshape_proj_x = Reshape(
        (input_shape[0], input_shape[2]), 
        name="reshape_proj_x"
    )(avg_pooling_2d_proj_x)
    
    reshape_proj_y = Reshape(
        (input_shape[1], input_shape[2]), 
        name="reshape_proj_y"
    )(avg_pooling_2d_proj_y)
    
    # NOW: Apply quantization to the 1D projections
    if use_delayed_quantization:
        quant_proj_x = SoftQuantizeLayer(
            n_bits=soft_quant_n_bits,
            initial_range=soft_quant_initial_range,
            trainable_levels=soft_quant_trainable_levels,
            trainable_bins=soft_quant_trainable_bins,
            initial_k=soft_quant_initial_k,
            trainable_k=soft_quant_trainable_k,
            name='soft_quantizer_output'  # Keep same name for scheduler
        )(reshape_proj_x)
        
        quant_proj_y = SoftQuantizeLayer(
            n_bits=soft_quant_n_bits,
            initial_range=soft_quant_initial_range,
            trainable_levels=soft_quant_trainable_levels,
            trainable_bins=soft_quant_trainable_bins,
            initial_k=soft_quant_initial_k,
            trainable_k=soft_quant_trainable_k,
            name='soft_quantizer_y'  # Different name for y projection
        )(reshape_proj_y)
    else:
        quant_proj_x = reshape_proj_x
        quant_proj_y = reshape_proj_y
    
    # Conv1D layers on quantized projections
    conv1d_proj_x = Conv1D(
        conv_filters, conv_kernel_size,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        bias_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="conv1d_proj_x"
    )(quant_proj_x)
    
    conv1d_proj_y = Conv1D(
        conv_filters, conv_kernel_size,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        bias_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="conv1d_proj_y"
    )(quant_proj_y)
    
    # Concatenate and continue with dense layers
    concatenate_layer = Concatenate(axis=1, name="concatenate")([conv1d_proj_x, conv1d_proj_y])
    activation_tanh_1 = Activation("tanh", name="activation_tanh_1")(concatenate_layer)
    
    flatten_layer = Flatten(name="flatten")(activation_tanh_1)
    
    # Dense layers
    dense_1 = Dense(
        dense1_units,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="dense_1"
    )(flatten_layer)
    activation_tanh_2 = Activation("tanh", name="activation_tanh_2")(dense_1)
    
    dense_2 = Dense(
        dense2_units,
        kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
        activity_regularizer=tf.keras.regularizers.L2(0.01),
        name="dense_2"
    )(activation_tanh_2)
    activation_tanh_3 = Activation("tanh", name="activation_tanh_3")(dense_2)
    
    # Output layer
    dense_3 = Dense(output_units, name="dense_3")(activation_tanh_3)
    
    model = Model(inputs=input_layer, outputs=dense_3, name="slim_delayed_quantization")
    return model


# Alternative: Quantize only one projection to test which one affects β
def CreateSlim1DModel_PartialQuantization(
    input_shape=(16, 16, 2),
    quantize_x_projection=True,  # Quantize X projection?
    quantize_y_projection=False,  # Quantize Y projection?
):
    """
    Test model where we only quantize one projection at a time
    to see which one affects β prediction
    """
    
    input_layer = Input(shape=input_shape, name="input_pxls/")
    
    # Average pooling
    avg_pooling_2d_proj_x = AveragePooling2D(
        pool_size=(1, input_shape[1])
    )(input_layer)
    
    avg_pooling_2d_proj_y = AveragePooling2D(
        pool_size=(input_shape[0], 1)
    )(input_layer)
    
    # Reshape
    reshape_proj_x = Reshape((input_shape[0], input_shape[2]))(avg_pooling_2d_proj_x)
    reshape_proj_y = Reshape((input_shape[1], input_shape[2]))(avg_pooling_2d_proj_y)
    
    # Selective quantization
    if quantize_x_projection:
        quant_proj_x = SoftQuantizeLayer(
            n_bits=2, initial_range=[-1.0, 1.0],
            trainable_levels=False, trainable_bins=True,
            initial_k=1.0, trainable_k=True,
            name='soft_quantizer_x'
        )(reshape_proj_x)
    else:
        quant_proj_x = reshape_proj_x
        
    if quantize_y_projection:
        quant_proj_y = SoftQuantizeLayer(
            n_bits=2, initial_range=[-1.0, 1.0],
            trainable_levels=False, trainable_bins=True,
            initial_k=1.0, trainable_k=True,
            name='soft_quantizer_y'
        )(reshape_proj_y)
    else:
        quant_proj_y = reshape_proj_y
    
    # Rest of the model
    conv1d_proj_x = Conv1D(5, 3)(quant_proj_x)
    conv1d_proj_y = Conv1D(5, 3)(quant_proj_y)
    
    concatenate_layer = Concatenate(axis=1)([conv1d_proj_x, conv1d_proj_y])
    activation_tanh_1 = Activation("tanh")(concatenate_layer)
    flatten_layer = Flatten()(activation_tanh_1)
    
    dense_1 = Dense(16, activation='tanh')(flatten_layer)
    dense_2 = Dense(16, activation='tanh')(dense_1)
    output = Dense(3)(dense_2)
    
    model = Model(inputs=input_layer, outputs=output)
    return model


# Usage examples for testing:

# 1. Test delayed quantization (most likely to work)
def test_delayed_quantization():
    model = CreateSlim1DModel_DelayedQuantization(
        use_delayed_quantization=True,
        soft_quant_n_bits=2,
        soft_quant_initial_k=1.0,
        soft_quant_trainable_k=True
    )
    return model

# 2. Test partial quantization to isolate the problem
def test_partial_quantization():
    # Test 1: Only quantize X projection
    model_x_only = CreateSlim1DModel_PartialQuantization(
        quantize_x_projection=True,
        quantize_y_projection=False
    )
    
    # Test 2: Only quantize Y projection  
    model_y_only = CreateSlim1DModel_PartialQuantization(
        quantize_x_projection=False,
        quantize_y_projection=True
    )
    
    return model_x_only, model_y_only

print("Models created! Test with:")
print("1. model = test_delayed_quantization()  # Most likely fix")
print("2. model_x, model_y = test_partial_quantization()  # For debugging")
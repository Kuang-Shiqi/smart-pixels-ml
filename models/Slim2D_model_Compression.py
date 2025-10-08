import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

# Try to import QKeras for quantization
try:
    from qkeras import *
    QKERAS_AVAILABLE = True
except ImportError:
    QKERAS_AVAILABLE = False
    print("Warning: QKeras not available. Quantization will be disabled.")

# Try to import TensorFlow Model Optimization for pruning
try:
    import tensorflow_model_optimization as tfmot
    PRUNING_AVAILABLE = True
except ImportError:
    PRUNING_AVAILABLE = False
    print("Warning: TensorFlow Model Optimization not available. Pruning will be disabled.")

def CreateSlim2DModel(
    input_shape=(16, 16, 2),
    # Model size parameters:
    conv_filters=5,        # Number of filters in Conv2D layers (currently 5)
    conv_kernel_size=3,    # Kernel size for Conv2D (currently 3)
    pool_size=2,           # Pool size for AveragePooling2D (currently varies)
    dense1_units=16,       # Units in first dense layer (currently 16)
    dense2_units=16,       # Units in second dense layer (currently 16)
    output_units=3,        # Output units (currently 3 for x, y, β)
    
    # Quantization parameters (matching your baseline model):
    use_quantization=False,    # Enable/disable quantization
    conv_weight_bits=4,        # Bits for Conv2D weights (4 like baseline)
    conv_activation_bits=4,    # Bits for Conv2D activations (4 like baseline)  
    dense_weight_bits=8,       # Bits for Dense weights (8 like baseline)
    dense_activation_bits=8,   # Bits for Dense activations (8 like baseline)
    
    # Pruning parameters:
    use_pruning=False,         # Enable/disable pruning
    pruning_fraction=0.5,      # Fraction of weights to prune (e.g., 0.5 = remove 50% of weights)
    pruning_schedule='constant' # 'constant' or 'polynomial_decay'
):
    """
    Conv2D architecture based on models.py with quantization and pruning options.
    
    Model size parameters:
    - conv_filters: Controls Conv2D layer width (5 → 3 or 2 for smaller)
    - conv_kernel_size: Controls Conv2D receptive field (3 → 2 for smaller)
    - pool_size: Controls AveragePooling2D downsampling (2 → 3 or 4 for more aggressive)
    - dense1_units: Controls first dense layer size (16 → 8 for smaller)
    - dense2_units: Controls second dense layer size (16 → 8 for smaller)
    - output_units: Should stay 3 for Slim model (x, y, β)
    
    Quantization parameters (matching your baseline model):
    - use_quantization: True/False to enable QKeras quantized layers
    - conv_weight_bits: 4-bit for Conv2D weights (like your baseline)
    - conv_activation_bits: 4-bit for Conv2D activations (like your baseline)
    - dense_weight_bits: 8-bit for Dense weights (like your baseline)  
    - dense_activation_bits: 8-bit for Dense activations (like your baseline)
    
    Pruning parameters:
    - use_pruning: True/False to enable structured pruning
    - pruning_fraction: Examples: 0.2=remove 20%, 0.5=remove 50%, 0.8=remove 80%
    - pruning_schedule: 'constant' or 'polynomial_decay'
    """
    
    # Validate inputs
    if use_quantization and not QKERAS_AVAILABLE:
        print("Warning: QKeras not available, falling back to non-quantized layers")
        use_quantization = False
        
    if use_pruning and not PRUNING_AVAILABLE:
        print("Warning: TensorFlow Model Optimization not available, pruning disabled")
        use_pruning = False
    
    # Helper function to create conv2d layers (following models.py conv_network pattern)
    def get_separable_conv2d_layer(filters, kernel_size, name):
        if use_quantization:
            return QSeparableConv2D(
                filters, kernel_size,
                depthwise_quantizer=quantized_bits(conv_weight_bits, 0, 1, alpha=1),
                pointwise_quantizer=quantized_bits(conv_weight_bits, 0, 1, alpha=1),
                bias_quantizer=quantized_bits(conv_weight_bits, 0, alpha=1),
                depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
                pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
                activity_regularizer=tf.keras.regularizers.L2(0.01),
                name=name
            )
        else:
            layer = SeparableConv2D(
                filters, kernel_size,
                depthwise_regularizer=tf.keras.regularizers.L1L2(0.01),
                pointwise_regularizer=tf.keras.regularizers.L1L2(0.01),
                activity_regularizer=tf.keras.regularizers.L2(0.01),
                name=name
            )
            
            # Apply pruning if enabled
            if use_pruning:
                pruning_params = {
                    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                        target_sparsity=pruning_fraction,
                        begin_step=0
                    ) if pruning_schedule == 'constant' else tfmot.sparsity.keras.PolynomialDecay(
                        initial_sparsity=0.0,
                        final_sparsity=pruning_fraction,
                        begin_step=100,
                        end_step=1000
                    )
                }
                layer = tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            
            return layer
    
    def get_conv2d_layer(filters, kernel_size, name):
        if use_quantization:
            return QConv2D(
                filters, kernel_size,
                kernel_quantizer=quantized_bits(conv_weight_bits, 0, alpha=1),
                bias_quantizer=quantized_bits(conv_weight_bits, 0, alpha=1),
                kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
                activity_regularizer=tf.keras.regularizers.L2(0.01),
                name=name
            )
        else:
            layer = Conv2D(
                filters, kernel_size,
                kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
                activity_regularizer=tf.keras.regularizers.L2(0.01),
                name=name
            )
            
            # Apply pruning if enabled
            if use_pruning:
                pruning_params = {
                    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                        target_sparsity=pruning_fraction,
                        begin_step=0
                    ) if pruning_schedule == 'constant' else tfmot.sparsity.keras.PolynomialDecay(
                        initial_sparsity=0.0,
                        final_sparsity=pruning_fraction,
                        begin_step=100,
                        end_step=1000
                    )
                }
                layer = tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            
            return layer
    
    def get_dense_layer(units, name):
        if use_quantization:
            return QDense(
                units,
                kernel_quantizer=quantized_bits(dense_weight_bits, 0, alpha=1),
                bias_quantizer=quantized_bits(dense_weight_bits, 0, alpha=1),
                kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
                activity_regularizer=tf.keras.regularizers.L2(0.01),
                name=name
            )
        else:
            layer = Dense(
                units,
                kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
                activity_regularizer=tf.keras.regularizers.L2(0.01),
                name=name
            )
            
            # Apply pruning if enabled
            if use_pruning:
                pruning_params = {
                    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(
                        target_sparsity=pruning_fraction,
                        begin_step=0
                    ) if pruning_schedule == 'constant' else tfmot.sparsity.keras.PolynomialDecay(
                        initial_sparsity=0.0,
                        final_sparsity=pruning_fraction,
                        begin_step=100,
                        end_step=1000
                    )
                }
                layer = tfmot.sparsity.keras.prune_low_magnitude(layer, **pruning_params)
            
            return layer
    
    def get_activation_layer(activation_name, name, is_conv_layer=False):
        if use_quantization:
            if is_conv_layer:
                # Use 4-bit for conv activations (like baseline)
                return QActivation(f"quantized_{activation_name}({conv_activation_bits}, 0, 1)", name=name)
            else:
                # Use 8-bit for dense activations (like baseline)
                return QActivation(f"quantized_{activation_name}({dense_activation_bits}, 0, 1)", name=name)
        else:
            return Activation(activation_name, name=name)
    
    def get_pooling_activation_layer(name):
        if use_quantization:
            return QActivation(f"quantized_bits({dense_activation_bits}, 0, alpha=1)", name=name)
        else:
            return Activation('linear', name=name)  # Identity activation for non-quantized
    
    # Build the model following models.py CreateModel structure
    input_layer = Input(shape=input_shape, name="input_pxls")
    
    # Conv network (following conv_network from models.py)
    # First layer: QSeparableConv2D
    conv_layer = get_separable_conv2d_layer(conv_filters, conv_kernel_size, "separable_conv2d")(input_layer)
    conv_activation_1 = get_activation_layer("tanh", "conv_activation_1", is_conv_layer=True)(conv_layer)
    
    # Second layer: QConv2D with kernel_size=1 (following models.py pattern)
    conv_layer_2 = get_conv2d_layer(conv_filters, 1, "conv2d")(conv_activation_1)
    conv_activation_2 = get_activation_layer("tanh", "conv_activation_2", is_conv_layer=True)(conv_layer_2)
    
    # AveragePooling2D (following CreateModel from models.py)
    pooling_layer = AveragePooling2D(
        pool_size=(pool_size, pool_size), 
        strides=None, 
        padding="valid", 
        data_format=None,
        name="avg_pooling_2d"
    )(conv_activation_2)
    
    # Post-pooling activation (following CreateModel from models.py)
    pooling_activation = get_pooling_activation_layer("pooling_activation")(pooling_layer)
    
    # Dense network (following var_network from models.py)
    flatten_layer = Flatten(name="flatten")(pooling_activation)
    
    # First dense layer
    dense_1 = get_dense_layer(dense1_units, "dense_1")(flatten_layer)
    dense_activation_1 = get_activation_layer("tanh", "dense_activation_1", is_conv_layer=False)(dense_1)
    
    # Second dense layer
    dense_2 = get_dense_layer(dense2_units, "dense_2")(dense_activation_1)
    dense_activation_2 = get_activation_layer("tanh", "dense_activation_2", is_conv_layer=False)(dense_2)
    
    # Output layer (usually not pruned, following var_network from models.py)
    if use_quantization:
        dense_3 = QDense(
            output_units,
            kernel_quantizer=quantized_bits(dense_weight_bits, 0, alpha=1),
            bias_quantizer=quantized_bits(dense_weight_bits, 0, alpha=1),
            kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
            name="dense_3"
        )(dense_activation_2)
    else:
        dense_3 = Dense(output_units, name="dense_3")(dense_activation_2)
    
    model = Model(inputs=input_layer, outputs=dense_3, name="smrtpxl_slim2d_tunable")
    return model


# Helper function for pruning training
def get_pruning_callbacks(model):
    """Returns callbacks needed for pruning training"""
    if not PRUNING_AVAILABLE:
        return []
    
    return [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir='./pruning_logs')
    ]

def finalize_pruned_model(model):
    """Strip pruning wrappers after training"""
    if not PRUNING_AVAILABLE:
        return model
    return tfmot.sparsity.keras.strip_pruning(model)


# Example usage with different configurations:
if __name__ == "__main__":
    
    # 1. Your original model (no compression)
    model_original = CreateSlim2DModel()
    print("Original model parameters:", model_original.count_params())
    
    # 2. Quantized model (4-bit conv, 8-bit dense - like your baseline)  
    model_quantized = CreateSlim2DModel(
        use_quantization=True,
        conv_weight_bits=4,     # Like your baseline
        dense_weight_bits=8     # Like your baseline  
    )
    print("Quantized model parameters:", model_quantized.count_params())
    
    # 3. Pruned model (remove 50% of weights)
    model_pruned = CreateSlim2DModel(
        use_pruning=True,
        pruning_fraction=0.5    # Remove 50% of weights
    )
    print("Pruned model parameters:", model_pruned.count_params())
    
    # 4. Combined: Small + Quantized + Pruned
    model_compressed = CreateSlim2DModel(
        conv_filters=3,           # Smaller architecture
        pool_size=3,              # More aggressive pooling
        dense1_units=8,
        dense2_units=8,
        use_quantization=True,    # 4-bit conv, 8-bit dense
        conv_weight_bits=4,
        dense_weight_bits=8, 
        use_pruning=True,         # Remove 70% of weights
        pruning_fraction=0.7
    )
    print("Fully compressed model parameters:", model_compressed.count_params())
    
    # 5. Ultra-aggressive compression
    model_ultra = CreateSlim2DModel(
        conv_filters=2,           # Very small
        conv_kernel_size=2,
        pool_size=4,              # Very aggressive pooling
        dense1_units=4,
        dense2_units=4,
        use_quantization=True,    # 4-bit conv, 8-bit dense
        conv_weight_bits=4,
        dense_weight_bits=8,
        use_pruning=True,         # Remove 80% of weights
        pruning_fraction=0.8
    )
    print("Ultra-compressed model parameters:", model_ultra.count_params())


# Training example with pruning:
def train_with_pruning_example():
    """
    Example of how to train a model with pruning enabled
    """
    # Create pruned model
    model = CreateSlim2DModel(
        use_pruning=True,
        pruning_fraction=0.6
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',  # Replace with your custom_loss
        metrics=['mae']
    )
    
    # Get pruning callbacks
    callbacks = get_pruning_callbacks(model)
    
    # Train model (replace with your actual training data)
    # history = model.fit(
    #     x_train, y_train,
    #     validation_data=(x_val, y_val),
    #     epochs=50,
    #     callbacks=callbacks
    # )
    
    # After training, strip pruning wrappers
    final_model = finalize_pruned_model(model)
    
    return final_model
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *

# Import soft quantization layer
from models.SoftQuantizeLayer import SoftQuantizeLayer

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

def CreateSlim1DModel(
    input_shape=(16, 16, 2),
    # Model size parameters:
    conv_filters=5,        # Number of filters in Conv1D layers (currently 5)
    conv_kernel_size=3,    # Kernel size for Conv1D (currently 3)
    dense1_units=16,       # Units in first dense layer (currently 16)
    dense2_units=16,       # Units in second dense layer (currently 16)
    output_units=3,        # Output units (currently 3 for x, y, β)
    
    # Quantization parameters (matching your baseline model):
    use_quantization=False,    # Enable/disable quantization
    conv_weight_bits=4,        # Bits for Conv1D weights (4 like baseline)
    conv_activation_bits=4,    # Bits for Conv1D activations (4 like baseline)  
    dense_weight_bits=8,       # Bits for Dense weights (8 like baseline)
    dense_activation_bits=8,   # Bits for Dense activations (8 like baseline)
    
    # Pruning parameters:
    use_pruning=False,         # Enable/disable pruning
    pruning_fraction=0.5,      # Fraction of weights to prune (e.g., 0.5 = remove 50% of weights)
    pruning_schedule='constant', # 'constant' or 'polynomial_decay'
    
    # ========== SOFT INPUT QUANTIZATION PARAMETERS ==========
    use_soft_input_quantization=False,  # Enable/disable soft input quantization
    soft_quant_n_bits=2,               # Number of quantization bits
    soft_quant_initial_range=[-1.0, 1.0],  # Initial quantization range
    soft_quant_trainable_levels=False,  # Whether quantization levels are trainable
    soft_quant_trainable_bins=True,     # Whether bin centers are trainable
    soft_quant_initial_k=1.0,          # Initial softness parameter
    soft_quant_trainable_k=True,       # Whether k parameter is trainable
):
    """
    Your exact working architecture with quantization, pruning, and soft input quantization options.
    
    Model size parameters:
    - conv_filters: Controls Conv1D layer width (5 → 3 or 2 for smaller)
    - conv_kernel_size: Controls Conv1D receptive field (3 → 2 for smaller) 
    - dense1_units: Controls first dense layer size (16 → 8 for smaller)
    - dense2_units: Controls second dense layer size (16 → 8 for smaller)
    - output_units: Should stay 3 for Slim model (x, y, β)
    
    Quantization parameters (matching your baseline model):
    - use_quantization: True/False to enable QKeras quantized layers
    - conv_weight_bits: 4-bit for Conv1D weights (like your baseline)
    - conv_activation_bits: 4-bit for Conv1D activations (like your baseline)
    - dense_weight_bits: 8-bit for Dense weights (like your baseline)  
    - dense_activation_bits: 8-bit for Dense activations (like your baseline)
    
    Pruning parameters:
    - use_pruning: True/False to enable structured pruning
    - pruning_fraction: Examples: 0.2=remove 20%, 0.5=remove 50%, 0.8=remove 80%
    - pruning_schedule: 'constant' or 'polynomial_decay'
    
    Soft Input Quantization parameters:
    - use_soft_input_quantization: True/False to enable soft input quantization
    - soft_quant_n_bits: Number of quantization bits (2 like Conv2D max model)
    - soft_quant_initial_range: Initial quantization range [-1.0, 1.0]
    - soft_quant_trainable_levels: Whether quantization levels are trainable
    - soft_quant_trainable_bins: Whether bin centers are trainable
    - soft_quant_initial_k: Initial softness parameter
    - soft_quant_trainable_k: Whether k parameter is trainable
    """
    
    # Validate inputs
    if use_quantization and not QKERAS_AVAILABLE:
        print("Warning: QKeras not available, falling back to non-quantized layers")
        use_quantization = False
        
    if use_pruning and not PRUNING_AVAILABLE:
        print("Warning: TensorFlow Model Optimization not available, pruning disabled")
        use_pruning = False
    
    # Helper function to create the right layer type
    def get_conv1d_layer(filters, kernel_size, name):
        if use_quantization:
            return QConv1D(
                filters, kernel_size,
                kernel_quantizer=quantized_bits(conv_weight_bits, 0, 1, alpha=1),  # 4-bit like baseline
                bias_quantizer=quantized_bits(conv_weight_bits, 0, 1, alpha=1),    # 4-bit like baseline
                kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
                bias_regularizer=tf.keras.regularizers.L1L2(0.01),
                activity_regularizer=tf.keras.regularizers.L2(0.01),
                name=name
            )
        else:
            layer = Conv1D(
                filters, kernel_size,
                kernel_regularizer=tf.keras.regularizers.L1L2(0.01),
                bias_regularizer=tf.keras.regularizers.L1L2(0.01),
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
                kernel_quantizer=quantized_bits(dense_weight_bits, 0, alpha=1),  # 8-bit like baseline
                bias_quantizer=quantized_bits(dense_weight_bits, 0, alpha=1),    # 8-bit like baseline
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
    
    # ========== MODEL ARCHITECTURE ==========
    
    # Input layer
    input_layer = Input(shape=input_shape, name="input_pxls/")
    
    # Apply soft input quantization if enabled (BEFORE average pooling)
    if use_soft_input_quantization:
        x_base = SoftQuantizeLayer(
            n_bits=soft_quant_n_bits,                     
            initial_range=soft_quant_initial_range,    
            trainable_levels=soft_quant_trainable_levels,        
            trainable_bins=soft_quant_trainable_bins,          
            initial_k=soft_quant_initial_k,                
            trainable_k=soft_quant_trainable_k,             
            name='soft_quantizer_output'  # Keep same name for AnnealingScheduler compatibility
        )(input_layer)
    else:
        x_base = input_layer
    
    # Average pooling operations (using the potentially quantized input)
    avg_pooling_2d_proj_x = AveragePooling2D(
        pool_size=(1, input_shape[1]), 
        name="avg_pooling_2d_proj_x"
    )(x_base)
    
    avg_pooling_2d_proj_y = AveragePooling2D(
        pool_size=(input_shape[0], 1), 
        name="avg_pooling_2d_proj_y"
    )(x_base)
    
    # Reshape operations
    reshape_proj_x = Reshape(
        (input_shape[0], input_shape[2]), 
        name="reshape_proj_x"
    )(avg_pooling_2d_proj_x)
    
    reshape_proj_y = Reshape(
        (input_shape[1], input_shape[2]), 
        name="reshape_proj_y"
    )(avg_pooling_2d_proj_y)
    
    # Conv1D layers with quantization/pruning support
    conv1d_proj_x = get_conv1d_layer(conv_filters, conv_kernel_size, "conv1d_proj_x")(reshape_proj_x)
    conv1d_proj_y = get_conv1d_layer(conv_filters, conv_kernel_size, "conv1d_proj_y")(reshape_proj_y)
    
    concatenate_layer = Concatenate(axis=1, name="concatenate")([conv1d_proj_x, conv1d_proj_y])
    activation_tanh_1 = get_activation_layer("tanh", "activation_tanh_1", is_conv_layer=True)(concatenate_layer)
    
    flatten_layer = Flatten(name="flatten")(activation_tanh_1)
    
    # Dense layers with quantization/pruning support
    dense_1 = get_dense_layer(dense1_units, "dense_1")(flatten_layer)
    activation_tanh_2 = get_activation_layer("tanh", "activation_tanh_2", is_conv_layer=False)(dense_1)
    
    dense_2 = get_dense_layer(dense2_units, "dense_2")(activation_tanh_2)
    activation_tanh_3 = get_activation_layer("tanh", "activation_tanh_3", is_conv_layer=False)(dense_2)
    
    # Output layer (usually not pruned)
    dense_3 = Dense(output_units, name="dense_3")(activation_tanh_3)
    
    model = Model(inputs=input_layer, outputs=dense_3, name="smrtpxl_slim_tunable")
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
    
    # 1. Your original model (no compression, no soft quantization)
    model_original = CreateSlim1DModel()
    print("Original model parameters:", model_original.count_params())
    
    # 2. Model with soft input quantization only
    model_soft_quant = CreateSlim1DModel(
        use_soft_input_quantization=True,
        soft_quant_n_bits=2,
        soft_quant_initial_range=[-1.0, 1.0]
    )
    print("Soft quantized input model parameters:", model_soft_quant.count_params())
    
    # 3. Quantized model (4-bit conv, 8-bit dense - like your baseline) + soft input quantization
    model_full_quantized = CreateSlim1DModel(
        use_quantization=True,
        conv_weight_bits=4,     # Like your baseline
        dense_weight_bits=8,    # Like your baseline  
        use_soft_input_quantization=True,  # Add soft input quantization
        soft_quant_n_bits=2
    )
    print("Fully quantized model parameters:", model_full_quantized.count_params())
    
    # 4. Combined: Small + Quantized + Pruned + Soft Input Quantization
    model_compressed = CreateSlim1DModel(
        conv_filters=3,           # Smaller architecture
        dense1_units=8,
        dense2_units=8,
        use_quantization=True,    # 4-bit conv, 8-bit dense
        conv_weight_bits=4,
        dense_weight_bits=8, 
        use_pruning=True,         # Remove 70% of weights
        pruning_fraction=0.7,
        use_soft_input_quantization=True,  # Add soft input quantization
        soft_quant_n_bits=2,
        soft_quant_initial_k=1.0,
        soft_quant_trainable_k=True
    )
    print("Fully compressed model parameters:", model_compressed.count_params())


# Training example with soft input quantization and annealing:
def train_with_soft_quantization_example():
    """
    Example of how to train a model with soft input quantization and annealing
    """
    # Create model with soft input quantization
    model = CreateSlim1DModel(
        use_soft_input_quantization=True,
        soft_quant_n_bits=2,
        soft_quant_initial_k=1.0,
        soft_quant_trainable_k=True,
        use_quantization=True,
        conv_weight_bits=4,
        dense_weight_bits=8
    )
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='mse',  # Replace with your custom_loss
        metrics=['mae']
    )
    
    # Import annealing scheduler (assuming it's available)
    # from models.AnnealingScheduler import AnnealingScheduler
    
    # Create annealing scheduler
    # scheduler_callback = AnnealingScheduler(
    #     schedule='cosine',  
    #     target_layer_name='soft_quantizer_output', 
    #     initial_k=1.0,
    #     final_k=67.0, 
    #     verbose=1      
    # )
    
    # Train model (replace with your actual training data)
    # history = model.fit(
    #     x_train, y_train,
    #     validation_data=(x_val, y_val),
    #     epochs=50,
    #     callbacks=[scheduler_callback]  # Add annealing scheduler
    # )
    
    return model
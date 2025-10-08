# Compact code to map normalized thresholds/levels to raw charges
import numpy as np

def map_normalized_to_raw_charge(model, data_generator):
    """Map soft quantization thresholds & levels from normalized to raw charge space"""
    
    # Extract soft quantization parameters
    sq_layer = model.get_layer(name="soft_quantizer_output")
    levels = sq_layer.bin_centers.numpy()  # The actual learned output levels
    
    # Calculate thresholds as midpoints between levels
    thresholds = [(levels[i] + levels[i+1]) / 2.0 for i in range(len(levels)-1)]
    thresholds = np.array(thresholds)
    
    # Get normalization parameters from data generator
    mu, sigma = data_generator.dataset_mean[0], data_generator.dataset_std[0] 
    s_pos, s_neg = data_generator.norm_factor_pos, data_generator.norm_factor_neg
    
    def reverse_transform(norm_vals):
        """Reverse the full normalization pipeline"""
        # Step 1: Reverse asymmetric scaling
        scaled_back = np.where(norm_vals > 0, norm_vals * s_pos, norm_vals * s_neg)
        
        # Step 2: Reverse standardization  
        log_vals = scaled_back * sigma + mu
        
        # Step 3: Reverse log compression: x_raw = sign(x_log) * (2^|x_log| - 1)
        raw_charges = np.sign(log_vals) * (2**np.abs(log_vals) - 1)
        
        return raw_charges
    
    # Apply reverse transformation
    raw_thresholds = reverse_transform(thresholds)
    raw_levels = reverse_transform(levels)
    
    # Print compact mapping
    print("NORMALIZED → RAW CHARGE MAPPING")
    print("=" * 40)
    
    print("QUANTIZATION THRESHOLDS:")
    for i, (norm, raw) in enumerate(zip(thresholds, raw_thresholds)):
        print(f"  Threshold {i+1}: {norm:8.3f} → {raw:10.1f}")
    
    print("\nQUANTIZATION LEVELS:")
    for i, (norm, raw) in enumerate(zip(levels, raw_levels)):
        print(f"  Level {i}: {norm:8.3f} → {raw:10.1f}")
    
    print(f"\nNormalization params: μ={mu:.3f}, σ={sigma:.3f}, s+={s_pos:.3f}, s-={s_neg:.3f}")
    
    return raw_thresholds, raw_levels

# Usage (assuming you have model and data_generator loaded):
# raw_thresholds, raw_levels = map_normalized_to_raw_charge(model, test_generator)
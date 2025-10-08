import tensorflow as tf
import tensorflow_probability as tfp

# custom loss function for Full model (8 outputs)
def custom_loss_full(y, p_base, minval=1e-9, maxval=1e9, scale=512):
    
    p = p_base
    
    # First 4 outputs: positions (x, y, cotAlpha, cotBeta)
    mu = p[:, 0:4]
    
    # Next 4 outputs: uncertainties (σ_x, σ_y, σ_α, σ_β)
    # Square them to get variances and ensure they're positive
    sigma = tf.abs(p[:, 4:8]) + minval  # This ensures all sigmas > 0
    Mdia = sigma * sigma  # Now we can square without worry

    # ADD THESE DEBUG PRINTS HERE:
    tf.print("Raw sigma predictions:", tf.reduce_mean(p[:, 4:8], axis=0))
    tf.print("Processed sigmas:", tf.reduce_mean(sigma, axis=0))
    tf.print("Variances (Mdia):", tf.reduce_mean(Mdia, axis=0))
    tf.print("Predicted means:", tf.reduce_mean(mu, axis=0))
    tf.print("Target means:", tf.reduce_mean(y, axis=0))
    tf.print("Residuals:", tf.reduce_mean(tf.abs(y - mu), axis=0))
    tf.print("---")
    zeros = tf.zeros_like(Mdia[:,0])
    
    # assembles diagonal scale_tril matrix (4x4 with only diagonal elements)
    row1 = tf.stack([tf.sqrt(Mdia[:,0]), zeros, zeros, zeros])
    row2 = tf.stack([zeros, tf.sqrt(Mdia[:,1]), zeros, zeros])
    row3 = tf.stack([zeros, zeros, tf.sqrt(Mdia[:,2]), zeros])
    row4 = tf.stack([zeros, zeros, zeros, tf.sqrt(Mdia[:,3])])

    scale_tril = tf.transpose(tf.stack([row1, row2, row3, row4]), perm=[2,0,1])

    dist = tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=scale_tril) 
    
    likelihood = dist.prob(y)  
    likelihood = tf.clip_by_value(likelihood, minval, maxval)

    NLL = -1*tf.math.log(likelihood)

    return tf.keras.backend.sum(NLL)
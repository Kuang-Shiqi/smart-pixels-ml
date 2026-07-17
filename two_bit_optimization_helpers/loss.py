import tensorflow as tf
import tensorflow_probability as tfp
import keras

# custom loss function
def custom_loss(y, p_base, minval=1e-9, maxval=1e9):
    
    p = p_base
    
    mu = p[:, 0:8:2]
    
    # creating each matrix element in 4x4
    Mdia = minval + tf.math.maximum(p[:, 1:8:2], 0.0)
    Mcov = p[:,8:]
    
    # placeholder zero element
    zeros = tf.zeros_like(Mdia[:,0])
    
    # assembles scale_tril matrix
    row1 = tf.stack([Mdia[:,0],zeros,zeros,zeros])
    row2 = tf.stack([Mcov[:,0],Mdia[:,1],zeros,zeros])
    row3 = tf.stack([Mcov[:,1],Mcov[:,2],Mdia[:,2],zeros])
    row4 = tf.stack([Mcov[:,3],Mcov[:,4],Mcov[:,5],Mdia[:,3]])

    scale_tril = tf.transpose(tf.stack([row1,row2,row3,row4]),perm=[2,0,1])

    dist = tfp.distributions.MultivariateNormalTriL(loc = mu, scale_tril = scale_tril) 
    
    likelihood = dist.prob(y)  
    likelihood = tf.clip_by_value(likelihood,minval,maxval)
    
    NLL = -1*tf.math.log(likelihood)

    return tf.keras.backend.sum(NLL)

# for FULL model (8 outputs)
def custom_diag_loss(y, p_base, minval=1e-9, maxval=1e9):
    mu = p_base[:, 0:4]
    
    raw_diag = p_base[:, 4:8]
    Mdia  = tf.nn.softplus(raw_diag) + minval
    
    dist = tfp.distributions.MultivariateNormalDiag(
        loc=mu,
        scale_diag=Mdia
    )
    NLL = -dist.log_prob(y)

    return tf.reduce_sum(NLL) 
    
# custom sse loss function for "slim" model
def custom_sse_loss(y, p_base, minval=1e-9, maxval=1e9):

    # truth values
    x_true = y[:,0]
    y_true = y[:,1]
    cotB_true = y[:,2]

    # predictions
    p = p_base
    x_pred = p[:,0]
    y_pred = p[:,1]
    cotB_pred = p[:,2]

    sse_x = tf.reduce_sum(tf.square(x_true - x_pred))
    sse_y = tf.reduce_sum(tf.square(y_true - y_pred))
    sse_cotB = tf.reduce_sum(tf.square(cotB_true - cotB_pred))

    return (sse_x + sse_y + sse_cotB) / 3.0
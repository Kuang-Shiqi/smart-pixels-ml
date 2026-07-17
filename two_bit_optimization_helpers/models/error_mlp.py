"""
Tiny projected MLP that outputs 10 Cholesky entries for the 4x4 covariance
matrix used by the Max-NLL loss. Mirrors the repo's x/y projection pattern
(mlp_encoder_model_nonquantized) but stripped to the minimal head:

  time-mean(input)              ->  (B, H, W, 1)
  AvgPool((1, W))               ->  x-profile (B, H)
  AvgPool((H, 1))               ->  y-profile (B, W)
  Concatenate                   ->  (B, H + W) = (B, 32)
  Dense(8, tanh)                ->  hidden       (32*8 + 8 = 264 params)
  Dense(10, linear)             ->  Cholesky     (8*10 + 10 = 90 params)

Total: 354 trainable params.

Same Cholesky parameterization as `loss.py:custom_loss` consumes (10 lower-
triangular entries; the loss applies softplus to the diagonal entries
internally).
"""
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Flatten, Dense, AveragePooling2D, Concatenate, Lambda,
)
from tensorflow.keras.models import Model


def build_error_mlp(shape=(16, 16, 2), hidden=8, n_outputs=10, name='error_mlp'):
    x_in = Input(shape=shape, name='cluster_input')
    x_t = Lambda(lambda t: tf.reduce_mean(t, axis=-1, keepdims=True),
                 name='time_mean')(x_in)
    proj_x = AveragePooling2D(pool_size=(1, shape[1]), name='avgpool_x')(x_t)
    proj_x = Flatten(name='flatten_x')(proj_x)
    proj_y = AveragePooling2D(pool_size=(shape[0], 1), name='avgpool_y')(x_t)
    proj_y = Flatten(name='flatten_y')(proj_y)
    h = Concatenate(name='concat_profiles')([proj_x, proj_y])
    h = Dense(hidden, activation='tanh', name='hidden')(h)
    out = Dense(n_outputs, activation='linear', name='chol_entries')(h)
    return Model(inputs=x_in, outputs=out, name=name)

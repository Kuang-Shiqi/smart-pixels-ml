# -*- coding: utf-8 -*-
# SoftQuantizeLayer.py
# @Author: Arghya Ranjan Das

import tensorflow as tf
import numpy as np
import math
from typing import Optional, List

# L_i = L0 + sum over j of softplus(Delta_L_raw[j])
# T_0 = T_off + softplus(Delta_T_raw[0])
# T_i = T_{i-1} + softplus(Delta_T_raw[i]) for i > 0
#
# `parametrization` selects how Delta is recovered from the raw weight:
#
#   'paper'  Delta = softplus(theta), theta_init = inv_softplus(Delta).
#            This is Eq. 2 of arXiv:2602.15946. d(Delta)/d(theta) = sigmoid(theta)
#            ~ 1, so every threshold moves at a comparable rate under a shared
#            learning rate.
#
#   'legacy' Delta = softplus(expm1(theta)), theta_init = log1p(Delta).
#            The historical form. It round-trips to the same *values* (softplus is
#            the identity for arguments of a few hundred electrons), so anything
#            trained with frozen thresholds is unaffected and existing checkpoints
#            still load. But d(Delta)/d(theta) ~ exp(theta) ~ Delta, which measures
#            ~200x larger threshold gradients than 'paper' at these charge scales:
#            the same learning rate that is right for the model weights sends the
#            thresholds flying. Kept as the default for checkpoint compatibility
#            only; use 'paper' for any training that unfreezes the thresholds.

class SoftQuantizeLayer(tf.keras.layers.Layer):
    """
    A soft quantization layer with fully trainable, non-uniform levels and bins.
    """
    def __init__(self,
                 n_bits: int=2,
                 initial_levels: Optional[List[float]]=None,
                 threshold_offset: float=0.0,
                 initial_thresholds: Optional[List[float]]=None,
                 trainable_levels=True,
                 trainable_thresholds=True,
                 initial_k=1.0,
                 trainable_k=False,
                 parametrization: str='legacy',
                 min_delta: float=1.0,
                 **kwargs):
        super(SoftQuantizeLayer, self).__init__(**kwargs)
        assert isinstance(n_bits, int) and n_bits > 0, "'n_bits' must be a positive integer."
        assert parametrization in ('paper', 'legacy'), \
            f"parametrization must be 'paper' or 'legacy', got {parametrization!r}"

        self.n_bits = n_bits
        self.num_levels = 2 ** self.n_bits

        self.initial_levels = initial_levels
        self.initial_thresholds = initial_thresholds
        self.threshold_offset = threshold_offset

        self.trainable_levels = trainable_levels
        self.trainable_thresholds = trainable_thresholds
        self.initial_k = initial_k
        self.trainable_k = trainable_k
        self.parametrization = parametrization
        self.min_delta = float(min_delta)

        self.initial_range = (-1.0, 1.0)
        
    @staticmethod
    def _softplus(z):
        return tf.nn.softplus(z)
    
    @staticmethod
    def _inv_softplus(z_positive, z_thr = 20.0):
        z = tf.convert_to_tensor(z_positive)                
        z = tf.cast(z, dtype=z.dtype)                           
        return tf.where(z > z_thr, z, tf.math.log(tf.math.expm1(z)))
    
    @staticmethod
    def _expm1(z, z_thr = 20.0):
        return tf.where(z > z_thr, tf.exp(z), tf.math.expm1(z))
    
    @staticmethod
    def _log1p(z):
        return tf.math.log1p(z)
    
    def _init_levels(self) -> np.ndarray:
        L = self.num_levels
        if self.initial_levels is not None:
            arr = np.asarray(self.initial_levels, dtype=np.float32)
            assert arr.shape[0] == L, "initial_levels length must equal 2^n_bits"
        else:
            lo, hi = self.initial_range
            arr = np.linspace(lo, hi, L, dtype=np.float32)
        if not np.all(np.diff(arr) > 0):
            arr = np.sort(arr)
        return arr  # (L,)
    
    def _init_thresholds(self):
        B = self.num_levels - 1
        if self.initial_thresholds is not None:
            arr = np.asarray(self.initial_thresholds, dtype=np.float32).astype(np.float32)
            assert arr.shape[0] == B, "initial_thresholds length must equal 2^n_bits - 1"
        else:
            lo, hi = self.initial_range
            arr = np.linspace(lo, hi, B, dtype=np.float32)
        if not np.all(np.diff(arr) > 0):
            arr = np.sort(arr)
        return arr  # (B,) = (L-1,)
    
    def build_levels(self):
        initial_levels = self._init_levels()

        if not self.trainable_levels:
            # Stored as numpy, converted to a tensor at call time. Two reasons:
            # overwriting self.initial_levels with a tensor made a second build()
            # -- which Keras 3 does routinely -- fail in _init_levels, and a
            # tf.constant created during build belongs to whatever scratch graph
            # build ran in, so it is out of scope by the time call() needs it.
            self._levels_np = np.asarray(initial_levels, dtype=np.float32)

        else:
            first_level_init = initial_levels[0]
            deltas_levels_init = np.diff(initial_levels)
            
            self.first_level = self.add_weight(
                name='first_level',
                shape=(1,),
                initializer=tf.constant_initializer(first_level_init),
                trainable=self.trainable_levels
            )

            level_deltas_raw_init = np.log1p(
                np.asarray(deltas_levels_init, dtype=np.float64)).astype(np.float32)
            self.level_deltas_raw = self.add_weight(
                name='level_deltas_raw',
                shape=(self.num_levels - 1,),
                initializer=tf.constant_initializer(level_deltas_raw_init),
                trainable=self.trainable_levels
            )


    def build_thresholds(self):
        B = self.num_levels - 1
        initial_thresholds = self._init_thresholds()
        # Deltas are: T0-T_off, T1-T0, T2-T1, ...
        deltas_thresholds_init = np.diff(initial_thresholds, 
                                         prepend=self.threshold_offset)

        assert np.all(deltas_thresholds_init > 0), f"\nInitial thresholds must be strictly increasing. \nGiven threshold_offset: {self.threshold_offset}, initial_thresholds: {initial_thresholds}\n Check if they satisfy: threshold_offset < T0 < T1 < ... < T{B-1}."

        # Initialisers are computed in numpy on purpose: build() can run inside a
        # symbolic tracing scope, where a tf op returns a SymbolicTensor and .numpy()
        # does not exist.
        if self.parametrization == 'paper':
            # Eq. 2 inverted: theta = inv_softplus(Delta - min_delta), guarded so a
            # 400 e- gap does not overflow expm1 in float32.
            assert np.all(deltas_thresholds_init > self.min_delta), (
                f"initial bin widths {deltas_thresholds_init} must all exceed "
                f"min_delta={self.min_delta}")
            z = np.asarray(deltas_thresholds_init - self.min_delta, dtype=np.float64)
            threshold_deltas_raw_init = np.where(
                z > 20.0, z, np.log(np.expm1(np.minimum(z, 20.0)))).astype(np.float32)
        else:
            threshold_deltas_raw_init = np.log1p(
                np.asarray(deltas_thresholds_init, dtype=np.float64)).astype(np.float32)
        self.threshold_deltas_raw = self.add_weight(
            name='threshold_deltas_raw',
            shape=(B,), 
            initializer=tf.constant_initializer(threshold_deltas_raw_init),
            trainable=self.trainable_thresholds
        )

    def build(self, input_shape):
        self.build_levels()
        self.build_thresholds()
        
        # --- parameter 'k' ---
        self.log_k = self.add_weight(
            name='log_k',
            shape=(1,),
            initializer=tf.constant_initializer(math.log(self.initial_k)),
            trainable=self.trainable_k
        )
        super(SoftQuantizeLayer, self).build(input_shape)

    @property
    def levels(self):
        """Calculates the trainable, non-uniform output levels."""
        if not self.trainable_levels:
            return tf.convert_to_tensor(self._levels_np, dtype=tf.float32)

        else:
            deltas = self._expm1(self.level_deltas_raw)
            # deltas = self._softplus(deltas)
            cumulative_deltas = tf.cumsum(deltas)
            return tf.concat([self.first_level, 
                            self.first_level + cumulative_deltas], 
                            axis=0
                            )
        
    def _threshold_deltas(self):
        """Strictly positive bin widths Delta_j. Monotonicity of T comes from
        these being positive by construction -- no sort, no clip, no penalty."""
        if self.parametrization == 'paper':
            # min_delta keeps bins strictly positive even if a delta is driven to
            # zero, so T stays strictly increasing and tau never collapses.
            return self.min_delta + self._softplus(self.threshold_deltas_raw)
        return self._softplus(self._expm1(self.threshold_deltas_raw))

    @property
    def thresholds(self):
        # Eq. 3: T_j = T_min + cumsum(Delta).  T_min is threshold_offset.
        return self.threshold_offset + tf.cumsum(self._threshold_deltas())

    @property
    def k(self):
        return tf.exp(self.log_k)

    @property
    def tau(self):
        # Eq. 5: tau_j = (Delta_j + Delta_{j+1}) / 2
        dT = self._threshold_deltas()
        right = tf.concat([dT[1:], dT[-1:]], axis=0)
        tau = 0.5 * (dT + right)                        
        tau = tf.maximum(tau, tf.cast(1e-6, tau.dtype))
        return tf.stop_gradient(tau)                                      
    
    def call(self, inputs, training=None):
        q_levels = self.levels
        q_thresholds = self.thresholds
        q_tau = self.tau
        hard_q = self._hard_quantize(inputs, q_levels, q_thresholds)
        
        if training:
            soft_q = self._soft_quantize(inputs, self.k, q_levels, q_thresholds, q_tau)
            return tf.stop_gradient(hard_q - soft_q) + soft_q
        return tf.stop_gradient(hard_q) 

    @staticmethod
    def _soft_quantize(x, k, levels, thresholds, tau):
        """
        Soft bin weights via smoothed CDF differences:
            weights = CDF_i - CDF_{i-1}, where CDF uses sigmoid(k*(t - x)).
        """
        x_exp = tf.expand_dims(x, axis=-1)                # (..., 1)
        # print(tau)
        sigs = tf.sigmoid(k * (thresholds - x_exp) / tau)       # (..., num_levels-1)
        left = tf.zeros_like(sigs[..., :1])               # (..., 1)
        right = tf.ones_like(sigs[..., :1])               # (..., 1)
        cdf = tf.concat([left, sigs, right], axis=-1)     # (..., B+2)
        weights = cdf[..., 1:] - cdf[..., :-1]            # (..., num_levels)
        return tf.reduce_sum(weights * levels, axis=-1)   # (...,)

    @staticmethod
    def _hard_quantize(x, levels, thresholds):
        x_reshaped = tf.expand_dims(x, axis=-1) 
        is_grt_th = x_reshaped > thresholds 
        indices = tf.reduce_sum(tf.cast(is_grt_th, dtype=tf.int32), axis=-1) 
        return tf.gather(levels, indices)

    def get_config(self):
        config = super(SoftQuantizeLayer, self).get_config()
        config.update({
            'n_bits': self.n_bits,
            'initial_range': self.initial_range,
            'trainable_levels': self.trainable_levels,
            'trainable_thresholds': self.trainable_thresholds,
            'initial_k': self.initial_k,
            'trainable_k': self.trainable_k,
            'parametrization': self.parametrization,
            'min_delta': self.min_delta,
        })
        return config
    
   

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    n_bits = 2
    num_levels = 2 ** n_bits
    B = num_levels - 1
    initial_k_val = 50.0
    initial_levels = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    threshold_offset = 80.0
    initital_thresholds = np.array([400.0, 800.0, 2000.0], dtype=np.float32)

    layer = SoftQuantizeLayer(
        n_bits=n_bits,
        initial_k=initial_k_val,
        initial_levels=initial_levels,
        threshold_offset=threshold_offset,
        initial_thresholds=initital_thresholds,
    )
    x_input = tf.constant(np.linspace(0, 2200, 3000), dtype=tf.float32)
    layer.build(input_shape=x_input.shape)

    # Current absolute values
    L_abs0 = layer.levels.numpy().astype(np.float32)      # (L,)
    T_abs0 = layer.thresholds.numpy().astype(np.float32)  # (B,)

    fig, ax = plt.subplots(figsize=(10, 10))
    plt.subplots_adjust(bottom=0.60)

    y_hard_initial = layer._hard_quantize(x_input, layer.levels, layer.thresholds)
    y_soft_initial = layer._soft_quantize(x_input, layer.k, layer.levels, layer.thresholds, layer.tau)

    (line_hard,) = ax.plot(x_input, y_hard_initial, 'r-', lw=2.5, label='Hard Quantize (Forward Pass)')
    (line_soft,) = ax.plot(x_input, y_soft_initial, 'b-', alpha=0.8, lw=2.0, label='Soft Quantize (Backprop Approx.)')

    vlines = ax.vlines(layer.thresholds.numpy(), -0.5, 4.0,
                       colors='g', lw=2, alpha=0.7, linestyles='--', label='Thresholds (T)')
    ax.vlines(layer.threshold_offset, -0.5, 4.0,
              colors='m', lw=2, alpha=0.7, linestyles='--', label='Threshold Offset (T_off)')

    ax.set_title(f"Interactive {n_bits}-bit Soft Quantizer", fontsize=16)
    ax.legend(loc='upper left')
    ax.grid(True)
    ax.set_xlim(0,2200)
    ax.set_ylim(-0.5, 4.0)

   

    plt.show()
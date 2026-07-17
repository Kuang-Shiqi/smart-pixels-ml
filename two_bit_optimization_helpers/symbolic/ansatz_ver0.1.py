"""
Physics ansatz layer: closed-form (x, y, |cot a|, |cot b|) estimator from a
16x16x2 pixel cluster, parameterized by a small set of trainable scalars.

Three variants, selected by the `variant` argument:
  - 'barycenter' : charge-weighted centroid + cluster-width angle estimator
                   (arxiv 2602.15946 eq 7, 10, 11; simplest and most stable)
  - 'localreco'  : head-tail position estimator (arxiv 2602.15946 eq 8, 9)
                   self-consistently fed by the barycenter angle estimate
  - 'blend'      : learned mix `(1-w) * barycenter + w * localreco` per output

Geometry constants come from the IOP paper (10.1088/2632-2153/ad6a00) appendix:
  p_x = 50 um  (pitch along x = axis-1 of the input tensor)
  p_y = 12.5 um (pitch along y = axis-2 of the input tensor)
  T   = 100 um (sensor thickness)

Axis convention follows the repo's existing MLP encoder
(mlp_encoder_model_nonquantized.py): AveragePooling2D((1,16)) gives the
x-projection, so axis 1 (H) maps to x and axis 2 (W) maps to y. Time slices
live on the channel axis (axis 3) and are summed before projection.

The layer outputs UNSIGNED `|cot a|` and `|cot b|` for the angle slots.
The student model wrapper is responsible for applying signs (e.g. borrowed
from the teacher). An affine head (per-output scale and bias, trainable)
absorbs residual calibration offsets.

Inputs:  charge tensor with shape (B, 16, 16, T)  with T = 2 by default
Outputs: tensor with shape (B, 4) -- order is (x, y, |cot a|, |cot b|)
"""
import tensorflow as tf
from tensorflow.keras import layers

GEOMETRY = dict(
    p_x=50.0,    # micrometers, x-pitch (axis-1 of input)
    p_y=12.5,    # micrometers, y-pitch (axis-2 of input)
    T=100.0,     # micrometers, sensor thickness
    N=16,        # array side in pixels
)


def _pixel_centers(n, pitch):
    """Pixel-center coordinates in microns, centered at zero."""
    idx = tf.range(n, dtype=tf.float32)
    return (idx - (n - 1) / 2.0) * pitch


def _safe_div(num, den, eps=1e-6):
    return num / (den + eps)


class PhysicsAnsatz(layers.Layer):
    def __init__(self, variant='barycenter',
                 fixed_pitch_x=None, fixed_pitch_y=None, fixed_T=None,
                 initial_theta_L_deg=22.0,
                 labels_scale=None,
                 **kwargs):
        """
        labels_scale: optional length-4 list of the per-output divisors used
            when normalizing TFRecord labels. When provided, the raw symbolic
            output is pre-divided by this so the output is already in label
            units; the affine head then sits near (scale=1, bias=0) at
            convergence. When omitted (default = ones) the layer behaves as
            in v1: the affine head must do all the scaling, which gradient
            descent solves by collapsing the symbolic to zero.
        """
        super().__init__(**kwargs)
        if variant not in ('barycenter', 'localreco', 'blend'):
            raise ValueError(f"unknown variant {variant!r}")
        self.variant = variant
        self.p_x = float(fixed_pitch_x if fixed_pitch_x is not None else GEOMETRY['p_x'])
        self.p_y = float(fixed_pitch_y if fixed_pitch_y is not None else GEOMETRY['p_y'])
        self.T = float(fixed_T if fixed_T is not None else GEOMETRY['T'])
        self.init_theta_L = float(initial_theta_L_deg) * 3.141592653589793 / 180.0
        if labels_scale is None:
            labels_scale = [1.0, 1.0, 1.0, 1.0]
        self.labels_scale_value = [float(v) for v in labels_scale]

    def build(self, input_shape):
        # constant (non-trainable) labels_scale tensor
        self.labels_scale = self.add_weight(
            name='labels_scale', shape=(4,),
            initializer=tf.constant_initializer(self.labels_scale_value),
            trainable=False,
        )
        # trainable Lorentz angle (radians)
        self.theta_L = self.add_weight(
            name='theta_L', shape=(),
            initializer=tf.constant_initializer(self.init_theta_L),
            trainable=True,
        )
        # trainable scalar in front of tan(theta_L) in the cot-beta branch
        # (Jennet's suggestion 2; init 1.0 = identity w.r.t. the bare formula)
        self.lorentz_scale = self.add_weight(
            name='lorentz_scale', shape=(),
            initializer=tf.constant_initializer(1.0),
            trainable=True,
        )
        # per-output affine (scale, bias) — 4 outputs
        self.aff_scale = self.add_weight(
            name='aff_scale', shape=(4,),
            initializer=tf.constant_initializer([1.0, 1.0, 1.0, 1.0]),
            trainable=True,
        )
        self.aff_bias = self.add_weight(
            name='aff_bias', shape=(4,),
            initializer=tf.constant_initializer([0.0, 0.0, 0.0, 0.0]),
            trainable=True,
        )
        if self.variant == 'blend':
            # blend logits, sigmoid -> [0,1] mixing weight per position output
            self.blend_logit_x = self.add_weight(
                name='blend_logit_x', shape=(),
                initializer=tf.constant_initializer(0.0), trainable=True,
            )
            self.blend_logit_y = self.add_weight(
                name='blend_logit_y', shape=(),
                initializer=tf.constant_initializer(0.0), trainable=True,
            )
        super().build(input_shape)

    def call(self, charge):
        # charge: (B, H=16, W=16, T=2). Sum over time -> (B, H, W).
        q = tf.reduce_sum(charge, axis=-1)
        # x-profile (sum over y / axis=2): shape (B, H)
        prof_x = tf.reduce_sum(q, axis=2)
        # y-profile (sum over x / axis=1): shape (B, W)
        prof_y = tf.reduce_sum(q, axis=1)

        N = GEOMETRY['N']
        x_centers = _pixel_centers(N, self.p_x)  # (N,)
        y_centers = _pixel_centers(N, self.p_y)

        # ---- barycenter positions ----
        sum_x = tf.reduce_sum(prof_x, axis=1)             # (B,)
        x_bary = _safe_div(tf.reduce_sum(prof_x * x_centers, axis=1), sum_x)
        sum_y = tf.reduce_sum(prof_y, axis=1)
        y_bary = _safe_div(tf.reduce_sum(prof_y * y_centers, axis=1), sum_y)

        # Lorentz drift offset on y
        dy = self.T * tf.tan(self.theta_L)
        y_bary_corr = y_bary + dy / 2.0

        # ---- cluster widths (non-differentiable count of nonzero pixels) ----
        active_x = tf.cast(prof_x > 0, tf.float32)
        active_y = tf.cast(prof_y > 0, tf.float32)
        w_x = tf.reduce_sum(active_x, axis=1)             # (B,)
        w_y = tf.reduce_sum(active_y, axis=1)

        # ---- cluster-width angle estimates (unsigned) ----
        # |cot a| = (w_x - 1) * p_x / T
        cota_abs = tf.maximum(w_x - 1.0, 0.0) * self.p_x / self.T
        # |cot b| = (w_y - 1) * p_y / T + |tan theta_L|  (Lorentz drift adds to y-cluster size)
        cotb_abs = tf.maximum(w_y - 1.0, 0.0) * self.p_y / self.T + tf.abs(self.lorentz_scale * tf.tan(self.theta_L))

        # ---- localreco head-tail positions (only computed if needed) ----
        if self.variant in ('localreco', 'blend'):
            # head/tail = first/last NONZERO pixel per profile.
            # Differentiable head/tail charges via mask * profile.
            # Find indices of first and last active pixels.
            # Use argmax of cumulative-active to get last active; first active
            # via argmax of active mask.
            # For batched per-event index lookup, use gather + range tricks.
            # Simplest: head = pixel at min idx where active=1; tail = max idx
            # where active=1. With ties (all zero), fallback to centroid.

            # Mask out zero rows entirely by setting -1 sentinel
            # First active index
            first_x = tf.argmax(active_x, axis=1, output_type=tf.int32)         # (B,)
            # Last active = N-1 - argmax(reversed_active)
            last_x = (N - 1) - tf.argmax(tf.reverse(active_x, axis=[1]),
                                         axis=1, output_type=tf.int32)
            first_y = tf.argmax(active_y, axis=1, output_type=tf.int32)
            last_y = (N - 1) - tf.argmax(tf.reverse(active_y, axis=[1]),
                                         axis=1, output_type=tf.int32)

            qF_x = tf.gather(prof_x, first_x, batch_dims=1)   # (B,)
            qL_x = tf.gather(prof_x, last_x, batch_dims=1)
            qF_y = tf.gather(prof_y, first_y, batch_dims=1)
            qL_y = tf.gather(prof_y, last_y, batch_dims=1)

            # Boundary x-coordinates between pixel pairs
            # x_F = position of boundary between first and (first+1)
            # x_L = position of boundary between (last-1) and last
            # boundary[k] = x_centers[k] + p_x/2  (between pixel k and k+1)
            xF = (tf.cast(first_x, tf.float32) - (N - 1) / 2.0) * self.p_x + self.p_x / 2.0
            xL = (tf.cast(last_x,  tf.float32) - (N - 1) / 2.0) * self.p_x - self.p_x / 2.0
            yF = (tf.cast(first_y, tf.float32) - (N - 1) / 2.0) * self.p_y + self.p_y / 2.0
            yL = (tf.cast(last_y,  tf.float32) - (N - 1) / 2.0) * self.p_y - self.p_y / 2.0

            # Eq 8: x = (qL-qF)/(qL+qF) * (T|cot a| - (xL - xF))/2 + (xF + xL)/2
            ratio_x = _safe_div(qL_x - qF_x, qL_x + qF_x)
            x_local = ratio_x * (self.T * cota_abs - (xL - xF)) / 2.0 + (xF + xL) / 2.0

            # Eq 9: y = ratio * (T|cot b + dy| - (yL - yF))/2 + (yF + yL)/2 + dy/2
            ratio_y = _safe_div(qL_y - qF_y, qL_y + qF_y)
            y_local = ratio_y * (self.T * cotb_abs - (yL - yF)) / 2.0 + (yF + yL) / 2.0 + dy / 2.0

        # ---- variant assembly ----
        if self.variant == 'barycenter':
            x_pred = x_bary
            y_pred = y_bary_corr
        elif self.variant == 'localreco':
            x_pred = x_local
            y_pred = y_local
        else:  # blend
            wx = tf.sigmoid(self.blend_logit_x)
            wy = tf.sigmoid(self.blend_logit_y)
            x_pred = (1.0 - wx) * x_bary + wx * x_local
            y_pred = (1.0 - wy) * y_bary_corr + wy * y_local

        raw = tf.stack([x_pred, y_pred, cota_abs, cotb_abs], axis=-1)   # (B, 4)
        # Pre-divide by labels_scale so raw lives in label units; the
        # affine head then only has to learn small corrections instead of
        # massive scale changes (which gradient descent prefers to do by
        # collapsing the symbolic).
        raw = raw / self.labels_scale
        return self.aff_scale * raw + self.aff_bias

"""
Physics ansatz layer: closed-form (x, y, SIGNED cot a, SIGNED cot b) estimator
from a 16x16xT pixel cluster, parameterized by a small set of trainable scalars.

Three variants, selected by the `variant` argument:
  - 'barycenter' : charge-weighted centroid + cluster-width angle estimator
                   (arxiv 2602.15946 eq 7, 10, 11; simplest and most stable)
  - 'localreco'  : head-tail position estimator (arxiv 2602.15946 eq 8, 9)
                   self-consistently fed by the barycenter angle estimate
  - 'blend'      : learned mix `(1-w) * barycenter + w * localreco` per output

Geometry constants come from the IOP paper (10.1088/2632-2153/ad6a00) appendix:
  p_x = 50 um  (pitch along x = axis-2 of the input tensor)
  p_y = 12.5 um (pitch along y = axis-1 of the input tensor)
  T   = 100 um (sensor thickness)

=========================== AXIS CONVENTION (v3, measured) ==========================
Axis 1 (rows) maps to y with the 12.5 um pitch; axis 2 (cols) maps to x with the
50 um pitch. Time slices live on the channel axis (axis 3).

This is the TRANSPOSE of what v2 of this file assumed. Measured directly from
50k centeredIncidence val events (axis_convention_diagnostic.ipynb):

    col centroid -> label x   corr 0.927, slope 43.2 um/px   (row -> x: corr 0.002)
    row centroid -> label y   corr 0.914, slope 10.5 um/px   (col -> y: corr 0.002)
    col width -1  -> |cot a|  corr 0.990, implied pitch 51.9 um
    row width -1  -> |cot b|  corr 0.944, implied pitch 14.8 um

(Centroid slopes read ~15% under the true pitch through ordinary regression
dilution; the ratio, 4.13 vs 50/12.5 = 4.00, is the scale-free check. The width
slopes are the cleaner pitch estimate.)

v2 patched only the two POSITION slots, from a wrapper outside this file. The
same two axes also carry the pitches, both angle magnitudes and the sign
observables, so the correction belongs here and applies to all of them at once.
Any external axis-swapping wrapper must now be REMOVED -- it would double-swap.

============================ SIGN (v2 TimeGrad, unchanged) ==========================
BOTH angles are emitted SIGNED, with NO external (teacher) sign needed.

Magnitude: cluster width, which is EVEN in the angle -> |cot|.
Sign: the between-time-slice centroid drift, which is ODD in the angle.
The July-2026 angle-sign study localized the sign to the (spatially-even x
time-odd) symmetry sector of the cluster and found single-scalar observables
that saturate the raw-512-pixel information ceiling (bal. acc ~0.96):

    Tx = <x>_{last slice} - <x>_{first slice}   (between-slice x-centroid drift, um)
    Ty = <y>_{last slice} - <y>_{first slice}   (between-slice y-centroid drift, um)

    sign(cot a) = tanh( sign_k_alpha * (sign_a0_alpha - Tx) )     [uncrossed:
    sign(cot b) = tanh( sign_k_beta  * (sign_a0_beta  - Ty) )      a<-Tx, b<-Ty]

The PAIRING here is byte-for-byte the same physical observable v2 used and is
measured at bal. acc 0.9661 (a) and 0.9644 (b); the other two pairings sit at
chance (0.504, 0.500). Only the NAMES change: what v2 called Ty was already the
column drift, i.e. the x-drift. Under the corrected axis names the coupling is
the uncrossed one the physics expects -- each angle is signed by the drift along
its OWN axis. Do not "re-cross" it back.

Sign-convention lock: k = -1 direction for both, i.e. tanh(k*(a0 - T)) with
k > 0. The drifts carry the corrected pitch, so the v2 calibration does not
transfer. Refitted on 40k train events under the v3 axes:
    k_alpha = 0.158 /um   a0_alpha = +0.16 um   (bal. acc 0.9672)
    k_beta  = 0.335 /um   a0_beta  = -10.86 um  (bal. acc 0.9631)
The defaults below are those values. Against T*tan(22 deg) = 40 um, a0_beta at
-10.9 um is not a clean match, so the v2 reading of a0_beta as "the Lorentz
E x B drift measured a second way" does not survive the pitch correction
-- flagged, not resolved. Both stay trainable, and the training notebook refits
(k, a0) from data by logistic fit, so these defaults are only a starting point.

The old y-skewness sign (chance-level, 0.50) is REMOVED. All time-summed spatial
moments are provably blind to the sign.

The training wrapper must NOT apply any external sign to either angle slot
(pass sa = sb = +1 to apply_sign, or skip it). Deployment: back up and replace
two_bit_optimization_helpers/symbolic/ansatz.py with this file, restart kernel.

Numerical hardening (same medicine as the old skew block): relu the charge
before the sign observables (analog pedestals < 0), eps-floored denominators,
finite-guard, and clipped tanh arguments. With T=1 time slice the drifts
degenerate to 0 and the sign becomes a trained constant (no crash).

Inputs:  charge tensor with shape (B, 16, 16, T)  with T = 2 by default
Outputs: tensor with shape (B, 4) -- order is (x, y, cot a[signed], cot b[signed])
"""
import tensorflow as tf
from tensorflow.keras import layers

GEOMETRY = dict(
    p_x=50.0,    # micrometers, x-pitch (axis-2 of input, i.e. columns)
    p_y=12.5,    # micrometers, y-pitch (axis-1 of input, i.e. rows)
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
                 initial_sign_k_alpha=0.158, initial_sign_a0_alpha=0.155,
                 initial_sign_k_beta=0.335,  initial_sign_a0_beta=-10.86,
                 **kwargs):
        """
        labels_scale: optional length-4 list of the per-output divisors used
            when normalizing TFRecord labels (raw output is pre-divided by it).
        initial_sign_k_*  : steepness of the tanh sign gate, in 1/um. Positive
            per the locked convention tanh(k * (a0 - T)); trainable, and CAN go
            negative if the data demands a convention flip.
        initial_sign_a0_* : sign-boundary offset in um. a0_beta absorbs the
            constant between-slice y-drift (~ -10.9 um measured; see the header
            on why the Lorentz reading of it is shaky).
            Best practice: calibrate both (k, a0) from a quick 1-feature
            logistic fit on the training data and pass them here.
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
        self.init_sign_k_alpha  = float(initial_sign_k_alpha)
        self.init_sign_a0_alpha = float(initial_sign_a0_alpha)
        self.init_sign_k_beta   = float(initial_sign_k_beta)
        self.init_sign_a0_beta  = float(initial_sign_a0_beta)

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
        self.lorentz_scale = self.add_weight(
            name='lorentz_scale', shape=(),
            initializer=tf.constant_initializer(1.0),
            trainable=True,
        )
        # ---- TimeGrad sign gates (NEW): sign = tanh(k * (a0 - drift)) ----
        self.sign_k_alpha = self.add_weight(
            name='sign_k_alpha', shape=(),
            initializer=tf.constant_initializer(self.init_sign_k_alpha),
            trainable=True,
        )
        self.sign_a0_alpha = self.add_weight(
            name='sign_a0_alpha', shape=(),
            initializer=tf.constant_initializer(self.init_sign_a0_alpha),
            trainable=True,
        )
        self.sign_k_beta = self.add_weight(
            name='sign_k_beta', shape=(),
            initializer=tf.constant_initializer(self.init_sign_k_beta),
            trainable=True,
        )
        self.sign_a0_beta = self.add_weight(
            name='sign_a0_beta', shape=(),
            initializer=tf.constant_initializer(self.init_sign_a0_beta),
            trainable=True,
        )
        # per-output affine (scale, bias) -- 4 outputs
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
            self.blend_logit_x = self.add_weight(
                name='blend_logit_x', shape=(),
                initializer=tf.constant_initializer(0.0), trainable=True,
            )
            self.blend_logit_y = self.add_weight(
                name='blend_logit_y', shape=(),
                initializer=tf.constant_initializer(0.0), trainable=True,
            )
        super().build(input_shape)

    # ---------------------------------------------------------------- sign --
    def _timegrad_signs(self, charge, x_centers, y_centers):
        """Between-slice centroid drifts Tx, Ty (um) -> soft signs in (-1, 1).

        Uses relu'd charge (analog pedestals < 0). A constant coordinate
        offset cancels in the first-minus-last difference, so centered pixel
        coordinates are equivalent to the study's corner-origin convention.
        """
        qs = tf.nn.relu(charge)                       # (B, 16, 16, T)
        q_first, q_last = qs[..., 0], qs[..., -1]     # (B, 16, 16)

        def _centroid(q2d, sum_axis, centers):
            # sum_axis=1 collapses rows -> col-indexed (x) profile;
            # sum_axis=2 collapses cols -> row-indexed (y) profile.
            prof = tf.reduce_sum(q2d, axis=sum_axis)                       # (B, 16)
            tot = tf.reduce_sum(prof, axis=1)                              # (B,)
            return _safe_div(tf.reduce_sum(prof * centers, axis=1), tot)

        # x lives on axis 2 (cols, 50 um); y on axis 1 (rows, 12.5 um).
        Tx = _centroid(q_last, 1, x_centers) - _centroid(q_first, 1, x_centers)
        Ty = _centroid(q_last, 2, y_centers) - _centroid(q_first, 2, y_centers)
        # finite-guard + physical clip (centroids bounded by the array extent)
        Tx = tf.clip_by_value(tf.where(tf.math.is_finite(Tx), Tx, tf.zeros_like(Tx)), -800.0, 800.0)
        Ty = tf.clip_by_value(tf.where(tf.math.is_finite(Ty), Ty, tf.zeros_like(Ty)), -200.0, 200.0)

        # Each angle is signed by the drift along its OWN axis (see header).
        arg_a = tf.clip_by_value(self.sign_k_alpha * (self.sign_a0_alpha - Tx), -30.0, 30.0)
        arg_b = tf.clip_by_value(self.sign_k_beta  * (self.sign_a0_beta  - Ty), -30.0, 30.0)
        return tf.tanh(arg_a), tf.tanh(arg_b)         # (B,), (B,)

    # ---------------------------------------------------------------- call --
    def call(self, charge):
        # charge: (B, H=16, W=16, T). Sum over time -> (B, H, W).
        q = tf.reduce_sum(charge, axis=-1)
        # x is the COLUMN axis (axis 2, 50 um); y is the ROW axis (axis 1, 12.5 um).
        prof_x = tf.reduce_sum(q, axis=1)   # x-profile, col-indexed: (B, W)
        prof_y = tf.reduce_sum(q, axis=2)   # y-profile, row-indexed: (B, H)

        N = GEOMETRY['N']
        x_centers = _pixel_centers(N, self.p_x)
        y_centers = _pixel_centers(N, self.p_y)

        # ---- barycenter positions ----
        sum_x = tf.reduce_sum(prof_x, axis=1)
        x_bary = _safe_div(tf.reduce_sum(prof_x * x_centers, axis=1), sum_x)
        sum_y = tf.reduce_sum(prof_y, axis=1)
        y_bary = _safe_div(tf.reduce_sum(prof_y * y_centers, axis=1), sum_y)

        # Lorentz drift offset on y. NOTE: for 'barycenter' this is a constant
        # added to one output, so it is fully degenerate with the trainable
        # aff_bias and cannot change the position fit. The non-degenerate
        # Lorentz term is the one inside cotb_abs below.
        dy = self.T * tf.tan(self.theta_L)
        y_bary_corr = y_bary + dy / 2.0

        # ---- TimeGrad signs for BOTH angles (replaces the old skew sign) ----
        sign_a, sign_b = self._timegrad_signs(charge, x_centers, y_centers)

        # ---- cluster widths ----
        active_x = tf.cast(prof_x > 0, tf.float32)
        active_y = tf.cast(prof_y > 0, tf.float32)
        w_x = tf.reduce_sum(active_x, axis=1)
        w_y = tf.reduce_sum(active_y, axis=1)

        # ---- cluster-width angle magnitudes (even in the angle) ----
        cota_abs = tf.maximum(w_x - 1.0, 0.0) * self.p_x / self.T
        cotb_abs = tf.maximum(w_y - 1.0, 0.0) * self.p_y / self.T \
                   + tf.abs(self.lorentz_scale * tf.tan(self.theta_L))
        # ---- apply the learned TimeGrad signs -> SIGNED angles ----
        cota_signed = cota_abs * sign_a
        cotb_signed = cotb_abs * sign_b

        # ---- localreco head-tail positions (magnitudes used internally) ----
        if self.variant in ('localreco', 'blend'):
            first_x = tf.argmax(active_x, axis=1, output_type=tf.int32)
            last_x = (N - 1) - tf.argmax(tf.reverse(active_x, axis=[1]),
                                         axis=1, output_type=tf.int32)
            first_y = tf.argmax(active_y, axis=1, output_type=tf.int32)
            last_y = (N - 1) - tf.argmax(tf.reverse(active_y, axis=[1]),
                                         axis=1, output_type=tf.int32)

            qF_x = tf.gather(prof_x, first_x, batch_dims=1)
            qL_x = tf.gather(prof_x, last_x, batch_dims=1)
            qF_y = tf.gather(prof_y, first_y, batch_dims=1)
            qL_y = tf.gather(prof_y, last_y, batch_dims=1)

            xF = (tf.cast(first_x, tf.float32) - (N - 1) / 2.0) * self.p_x + self.p_x / 2.0
            xL = (tf.cast(last_x,  tf.float32) - (N - 1) / 2.0) * self.p_x - self.p_x / 2.0
            yF = (tf.cast(first_y, tf.float32) - (N - 1) / 2.0) * self.p_y + self.p_y / 2.0
            yL = (tf.cast(last_y,  tf.float32) - (N - 1) / 2.0) * self.p_y - self.p_y / 2.0

            ratio_x = _safe_div(qL_x - qF_x, qL_x + qF_x)
            x_local = ratio_x * (self.T * cota_abs - (xL - xF)) / 2.0 + (xF + xL) / 2.0
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

        # slots 2 AND 3 are now SIGNED.
        raw = tf.stack([x_pred, y_pred, cota_signed, cotb_signed], axis=-1)
        raw = raw / self.labels_scale
        return self.aff_scale * raw + self.aff_bias

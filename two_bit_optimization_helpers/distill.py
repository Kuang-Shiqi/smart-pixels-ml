"""
Distillation training loop. Frozen teacher (e.g. Conv2D_Max or ViT_Max)
supervises a tiny student (PhysicsAnsatz + 354-param error MLP) under

    L_total = L_data + lambda * KL[ N_teacher || N_student ]

with a Modified Differential Method of Multipliers (MDMM) update on
lambda each step:

    lambda <- max(0, lambda + eta * (KL_batch - kl_target))

KL is the closed-form Gaussian KL via tfp.MultivariateNormalTriL.
L_data is the same `custom_loss` (Max NLL) the repo already uses.
"""
import tensorflow as tf
import tensorflow_probability as tfp

from loss import custom_loss
from models.student_max import pack_14, apply_sign, teacher_signs

MINVAL = 1e-9


def make_tril(v14, minval=MINVAL):
    """Slice a (B, 14) tensor (custom_loss layout) into mean (B,4) and
    lower-triangular Cholesky factor (B, 4, 4).

    Uses softplus on the diagonal entries to guarantee strict positivity
    for KL stability (custom_loss internally uses max(0,.); we keep the
    same raw entries but apply a smoother positivity transform here so
    near-zero raw values don't produce singular covariances during KL).
    """
    mu = v14[..., 0:8:2]
    diag = minval + tf.nn.softplus(v14[..., 1:8:2])
    off = v14[..., 8:14]
    z = tf.zeros_like(diag[..., 0])
    row1 = tf.stack([diag[..., 0], z, z, z], axis=-1)
    row2 = tf.stack([off[..., 0], diag[..., 1], z, z], axis=-1)
    row3 = tf.stack([off[..., 1], off[..., 2], diag[..., 2], z], axis=-1)
    row4 = tf.stack([off[..., 3], off[..., 4], off[..., 5], diag[..., 3]], axis=-1)
    L = tf.stack([row1, row2, row3, row4], axis=-2)
    return mu, L


def gaussian_kl(teacher_14, student_14):
    """Mean KL[ N(mu_T, Sigma_T) || N(mu_S, Sigma_S) ] over the batch."""
    mu_T, L_T = make_tril(teacher_14)
    mu_S, L_S = make_tril(student_14)
    d_T = tfp.distributions.MultivariateNormalTriL(loc=mu_T, scale_tril=L_T)
    d_S = tfp.distributions.MultivariateNormalTriL(loc=mu_S, scale_tril=L_S)
    return tf.reduce_mean(tfp.distributions.kl_divergence(d_T, d_S))


class Distiller(tf.keras.Model):
    """Wraps a two-headed student + frozen teacher into a fit()-compatible
    distillation model. Per-step MDMM update on lambda."""

    def __init__(self, student, teacher,
                 lambda_init=0.0, mdmm_eta=1e-3, kl_target=0.0,
                 warmup_steps=0, **kwargs):
        super().__init__(**kwargs)
        self.student = student
        self.teacher = teacher
        for w in self.teacher.weights:
            w._trainable = False
        self.teacher.trainable = False

        self.lam = tf.Variable(lambda_init, dtype=tf.float32,
                               trainable=False, name='mdmm_lambda')
        self.mdmm_eta = float(mdmm_eta)
        self.kl_target = float(kl_target)
        self.warmup_steps = int(warmup_steps)
        self._step = tf.Variable(0, dtype=tf.int64, trainable=False, name='train_step')

        self.loss_data_tracker = tf.keras.metrics.Mean(name='loss_data')
        self.kl_tracker = tf.keras.metrics.Mean(name='kl')
        self.lam_tracker = tf.keras.metrics.Mean(name='lam')

    def call(self, x, training=False):
        return self.student(x, training=training)

    def _forward(self, x, training):
        t14 = self.teacher(x, training=False)
        sa, sb = teacher_signs(t14)
        means_uns, chol = self.student(x, training=training)
        s14 = pack_14(apply_sign(means_uns, sa, sb), chol)
        return t14, s14

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            t14, s14 = self._forward(x, training=True)
            n = tf.cast(tf.shape(x)[0], tf.float32)
            data_loss = custom_loss(y, s14) / n
            kl = gaussian_kl(t14, s14)
            warm = tf.cast(self._step >= self.warmup_steps, tf.float32)
            total = data_loss + warm * self.lam * kl
        grads = tape.gradient(total, self.student.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_weights))
        # MDMM dual update, with per-step cap to prevent runaway when KL
        # is very large early in training.
        delta = tf.clip_by_value(self.mdmm_eta * (kl - self.kl_target), -1.0, 1.0)
        self.lam.assign(tf.maximum(0.0, self.lam + delta))
        self._step.assign_add(1)
        self.loss_data_tracker.update_state(data_loss)
        self.kl_tracker.update_state(kl)
        self.lam_tracker.update_state(self.lam)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        t14, s14 = self._forward(x, training=False)
        n = tf.cast(tf.shape(x)[0], tf.float32)
        data_loss = custom_loss(y, s14) / n
        kl = gaussian_kl(t14, s14)
        self.loss_data_tracker.update_state(data_loss)
        self.kl_tracker.update_state(kl)
        return {'loss_data': self.loss_data_tracker.result(),
                'kl': self.kl_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_data_tracker, self.kl_tracker, self.lam_tracker]

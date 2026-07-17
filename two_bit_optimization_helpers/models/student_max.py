"""
Student model for the symbolic + tiny-NN Max-NLL distillation.

The student is a two-headed Keras Model:
  input  : cluster   (B, 16, 16, T=2)
  outputs:
    sym_means_unsigned (B, 4)  -- (x, y, |cot a|, |cot b|) from PhysicsAnsatz
    chol_entries       (B, 10) -- raw Cholesky entries from the tiny MLP

The training loop is responsible for:
  1. Applying the per-event sign on the cot a / cot b means (signs borrowed
     from the frozen teacher's prediction during distillation).
  2. Packing the means + Cholesky into the 14-vector that `custom_loss`
     expects (mean, diag, mean, diag, ... ; then 6 off-diagonals).

A `pack_14` helper does the assembly so both the data NLL and the KL term
share the same construction.
"""
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

from symbolic import PhysicsAnsatz
from models.error_mlp import build_error_mlp


def build_student_max(variant='barycenter', shape=(16, 16, 2),
                      ansatz_kwargs=None, name=None):
    """Two-headed student model: cluster -> (unsigned means (B,4), chol (B,10))."""
    ansatz_kwargs = dict(ansatz_kwargs or {})
    name = name or f'student_max_{variant}'
    cluster = Input(shape=shape, name='cluster')
    sym = PhysicsAnsatz(variant=variant, name='physics_ansatz',
                        **ansatz_kwargs)(cluster)
    err = build_error_mlp(shape=shape, name='error_mlp')
    chol = err(cluster)
    return Model(inputs=cluster, outputs=[sym, chol], name=name)


def pack_14(means_signed, chol, minval=1e-9):
    """Interleave (means, diagonals) and append off-diagonals to match
    the 14-output layout `loss.custom_loss` consumes.

    means_signed : (B, 4) -- [x, y, cot_a, cot_b] with signs already applied
    chol         : (B, 10)
       chol[..., 0..3]  -> raw diagonal entries (loss clips/softplus internally)
       chol[..., 4..9]  -> off-diagonal Cholesky entries (M21, M31, M32, M41, M42, M43)
    """
    mu = means_signed
    diag = chol[..., :4]
    off = chol[..., 4:]
    out = tf.stack([mu[..., 0], diag[..., 0],
                    mu[..., 1], diag[..., 1],
                    mu[..., 2], diag[..., 2],
                    mu[..., 3], diag[..., 3]], axis=-1)   # (B, 8)
    return tf.concat([out, off], axis=-1)                  # (B, 14)


def apply_sign(means_unsigned, sign_a, sign_b):
    """Apply per-event signs to the cot a / cot b slots of the symbolic means."""
    return tf.stack([
        means_unsigned[..., 0],
        means_unsigned[..., 1],
        means_unsigned[..., 2] * sign_a,
        means_unsigned[..., 3] * sign_b,
    ], axis=-1)


def teacher_signs(teacher_14):
    """Extract (sign_a, sign_b) from a teacher's 14-output prediction.

    `loss.custom_loss` reads means at indices 0, 2, 4, 6 -> cot a is at 4, cot b at 6.
    """
    one = tf.ones_like(teacher_14[..., 4])
    s_a = tf.sign(teacher_14[..., 4])
    s_b = tf.sign(teacher_14[..., 6])
    s_a = tf.where(tf.equal(s_a, 0.0), one, s_a)
    s_b = tf.where(tf.equal(s_b, 0.0), one, s_b)
    return s_a, s_b

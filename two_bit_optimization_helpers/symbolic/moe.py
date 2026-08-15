"""Model assembly for the symbolic student: DigitizeLayer -> router + experts.

These classes used to live inside the training script, which meant a saved
`.weights.h5` could not be reloaded without copying the script. They are here so
inference, export and hardware work can rebuild a trained arm from
`summary.json` + weights alone:

    from symbolic.moe import load_arm
    model, core, meta = load_arm("/depot/.../digi_2bit_paper_code_nexp4")
    y14 = model(x).numpy()          # (B, 14)

Layer names and attribute paths are load-bearing -- they are how Keras maps
weights onto the graph. Do not rename anything here without re-checking that the
existing runs still load (scripts/export_for_hardware.py verifies exactly that,
by replaying a run's own parquet).
"""
import json
import os

import numpy as np
import tensorflow as tf

from symbolic.digitize import DigitizeLayer
from models.student_max import build_student_max, pack_14


class RouterFeatures(tf.keras.layers.Layer):
    """36 features from the digitized cluster: x/y profiles + 4 shape scalars.

    HARDENED: relu charge, std floored at 1.0 (m3/std**3 underflows for narrow
    clusters otherwise), finite-guard and clip.
    """

    def call(self, x):
        xp = tf.nn.relu(x)
        q = tf.reduce_sum(xp, axis=-1)
        px = tf.reduce_sum(q, axis=2)
        py = tf.reduce_sum(q, axis=1)
        tot = tf.reduce_sum(py, axis=1, keepdims=True) + 1e-9
        pxn, pyn = px / tot, py / tot
        yc = tf.range(16, dtype=tf.float32) - 7.5
        mu = tf.reduce_sum(pyn * yc, axis=1, keepdims=True)
        d = yc[None, :] - mu
        m2 = tf.reduce_sum(pyn * d ** 2, axis=1, keepdims=True)
        m3 = tf.reduce_sum(pyn * d ** 3, axis=1, keepdims=True)
        std = tf.sqrt(m2 + 1.0)
        skew = m3 / (std ** 3)
        py0 = tf.reduce_sum(xp[..., 0], axis=1)
        py1 = tf.reduce_sum(xp[..., -1], axis=1)
        c0 = tf.reduce_sum(py0 * yc, 1, keepdims=True) / (tf.reduce_sum(py0, 1, keepdims=True) + 1e-9)
        c1 = tf.reduce_sum(py1 * yc, 1, keepdims=True) / (tf.reduce_sum(py1, 1, keepdims=True) + 1e-9)
        f = tf.concat([pxn, pyn, std, skew, c1 - c0, tf.math.log(tot + 1.0)], axis=-1)
        f = tf.where(tf.math.is_finite(f), f, tf.zeros_like(f))
        return tf.clip_by_value(f, -8.0, 8.0)


class MoECore(tf.keras.Model):
    """N experts, softmax router. The input is digitized ONCE and shared."""

    def __init__(self, n_experts, router_hidden, temp, make_digitizer, make_expert, **kw):
        super().__init__(**kw)
        self.n_experts, self.temp = n_experts, temp
        self.digi = make_digitizer()
        self.feats = RouterFeatures(name="router_features")
        self.router = tf.keras.Sequential(
            [tf.keras.layers.Dense(h, activation="relu") for h in router_hidden]
            + [tf.keras.layers.Dense(n_experts)], name="router")
        self.experts = [make_expert() for _ in range(n_experts)]

    def _route_d(self, xd):
        return tf.nn.softmax(self.router(self.feats(xd)) / self.temp, axis=-1)

    def route(self, x):
        return self._route_d(self.digi(x, training=False))

    def call(self, x, training=False):
        xd = self.digi(x, training=training)
        w = self._route_d(xd)
        outs = [e(xd, training=training) for e in self.experts]
        means = tf.add_n([w[:, k:k + 1] * outs[k][0] for k in range(self.n_experts)])
        chol = tf.add_n([w[:, k:k + 1] * outs[k][1] for k in range(self.n_experts)])
        return means, chol


class SingleCore(tf.keras.Model):
    """N_EXPERTS == 1: one closed-form expert, no router at all."""

    def __init__(self, make_digitizer, make_expert, **kw):
        super().__init__(**kw)
        self.digi = make_digitizer()
        self.experts = [make_expert()]

    def call(self, x, training=False):
        return self.experts[0](self.digi(x, training=training), training=training)


class PackedStudent(tf.keras.Model):
    """(mu, chol) -> the packed 14-vector the loss and the parquet dump use."""

    def __init__(self, core, **kw):
        super().__init__(**kw)
        self.core = core

    def call(self, x, training=False):
        mu, chol = self.core(x, training=training)
        return pack_14(mu, chol)


def build_student(n_bits, thresholds, levels, threshold_offset, labels_scale, sign_init,
                  n_experts=1, router_hidden=(), router_temp=1.0, variant="barycenter",
                  ste=True, trainable_thresholds=False, initial_k=50.0,
                  parametrization="paper"):
    """Assemble the full student. Returns (model, core)."""
    def make_digitizer():
        return DigitizeLayer(n_bits=n_bits, thresholds=thresholds, levels=levels,
                             threshold_offset=threshold_offset, ste=ste,
                             trainable_thresholds=trainable_thresholds,
                             trainable_levels=False, initial_k=initial_k,
                             parametrization=parametrization, name="digitize")

    def make_expert():
        return build_student_max(variant,
                                 ansatz_kwargs={"labels_scale": list(labels_scale),
                                                **dict(sign_init)})

    core = (SingleCore(make_digitizer, make_expert, name="single_core") if n_experts == 1
            else MoECore(n_experts, list(router_hidden), router_temp,
                         make_digitizer, make_expert, name="moe_core"))
    return PackedStudent(core, name=f"symbolic_n{n_experts}"), core


def load_arm(run_dir, weights=None):
    """Rebuild a trained arm from its run directory. Returns (model, core, summary).

    Everything needed is in summary.json: bit depth, the thresholds AS TRAINED
    (thresholds_final_e, which differ from the initial ones on a trainable-threshold
    arm), the levels, the sign-gate init and the router shape.
    """
    summary = json.load(open(os.path.join(run_dir, "summary.json")))
    g = summary.get("digitization") or {}
    n_experts = int(summary["n_experts"])
    n_bits = g.get("n_bits")

    # thresholds_final_e is what the ADC ended up as; that is what inference must use
    thr = g.get("thresholds_final_e", g.get("thresholds"))
    lv = g.get("levels_final", g.get("levels"))
    if n_bits is not None:
        thr = np.asarray(thr, np.float32)
        lv = (np.arange(2 ** n_bits, dtype=np.float32) if lv is None
              else np.asarray(lv, np.float32))

    ls = summary.get("labels_scale")
    labels_scale = (ls["train"] if isinstance(ls, dict) else ls)
    if labels_scale is None:                       # runs predating the labels_scale block
        labels_scale = json.load(open(os.path.join(run_dir, "labels_scale.json")))["labels_scale"]

    model, core = build_student(
        n_bits=n_bits, thresholds=thr, levels=lv,
        threshold_offset=g.get("threshold_offset_e", 80.0),
        labels_scale=labels_scale, sign_init=summary.get("sign_init", {}),
        n_experts=n_experts, router_hidden=summary.get("router_hidden", ()),
        router_temp=summary.get("router_temp", 1.0),
        variant=summary.get("variant", "barycenter"),
        initial_k=g.get("final_k", g.get("initial_k", 50.0)))

    model(np.zeros((1, 16, 16, 2), np.float32))    # build the graph before loading
    model.load_weights(weights or os.path.join(run_dir, f"symbolic_n{n_experts}.weights.h5"))
    return model, core, summary

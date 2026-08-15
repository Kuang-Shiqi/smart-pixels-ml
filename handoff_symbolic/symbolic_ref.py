"""Pure-numpy forward pass of the SmartPixels symbolic student.

No TensorFlow, no Keras, no qkeras -- just numpy and a JSON of constants. This is
the transliteration target for C++/HLS and the oracle to test it against:
`verify.py` checks it against the Keras model bit-for-bit on the golden vectors.

    import json, numpy as np
    from symbolic_ref import forward, digitize
    C = json.load(open("exports/nexp1/constants.json"))
    codes = digitize(analog_charge, C)      # (B,16,16,2) -> 2-bit codes 0..3
    y14   = forward(codes, C)               # -> (B,14)

Everything is float32 and vectorized over the batch, but nothing here needs a
batch: every event is independent, and the per-event op count is in README.md.

LAYOUT
------
input   (16, 16, 2) : [row, col, time_slice].  row -> y @ 12.5 um pitch,
                                               col -> x @ 50 um pitch.
        The row/col convention is measured, not assumed -- see README.
output  (14,)       : [x, sx, y, sy, cotA, sA, cotB, sB, M21, M31, M32, M41, M42, M43]
        means at 0,2,4,6; raw Cholesky diagonals at 1,3,5,7 (apply softplus to get
        sigma); 6 off-diagonal Cholesky entries last.
"""
import numpy as np

F32 = np.float32


# ------------------------------------------------------------------ ADC --
def digitize(charge, C):
    """Analog charge in electrons -> 2-bit code 0..3. Hard quantizer, STRICTLY >.

    Skip this if your hardware receives ADC codes directly from the sensor -- the
    thresholds are the on-chip comparators, not part of the network.
    """
    d = C["digitize"]
    if d.get("n_bits") is None:
        return np.asarray(charge, F32)
    thr = np.asarray(d["thresholds_e"], F32)
    lv = np.asarray(d["levels"], F32)
    idx = (np.asarray(charge, F32)[..., None] > thr).sum(-1)
    return lv[idx].astype(F32)


# -------------------------------------------------------------- physics --
def _centroid(prof, centers):
    """charge-weighted mean; the +1e-6 matches the trained graph's _safe_div"""
    return (prof * centers).sum(-1) / (prof.sum(-1) + F32(1e-6))


def ansatz(qd, E, C):
    """One expert: digitized cluster (B,16,16,2) -> 4 means (B,4).

    'barycenter' variant. 18 trainable scalars, all folded into E.
    """
    g = C["geometry"]
    xc = ((np.arange(g["N"], dtype=F32) - (g["N"] - 1) / 2.0) * g["p_x"]).astype(F32)
    yc = ((np.arange(g["N"], dtype=F32) - (g["N"] - 1) / 2.0) * g["p_y"]).astype(F32)

    q = qd.sum(-1)                       # sum over time  -> (B,16,16)
    prof_x = q.sum(1)                    # collapse rows  -> col-indexed = x
    prof_y = q.sum(2)                    # collapse cols  -> row-indexed = y

    # --- positions: charge-weighted barycenters ---
    x_pred = _centroid(prof_x, xc)
    y_pred = _centroid(prof_y, yc) + F32(E["dy_over_2"])       # Lorentz offset (constant)

    # --- angle magnitudes: cluster width is EVEN in the angle, so |cot| only ---
    w_x = (prof_x > 0).sum(-1).astype(F32)
    w_y = (prof_y > 0).sum(-1).astype(F32)
    cota = np.maximum(w_x - 1.0, 0.0) * F32(g["p_x"] / g["T"])
    cotb = np.maximum(w_y - 1.0, 0.0) * F32(g["p_y"] / g["T"]) + F32(E["lorentz_term"])

    # --- sign: between-slice centroid DRIFT (the only sign-carrying observable) ---
    qs = np.maximum(qd, 0.0)
    q0, q1 = qs[..., 0], qs[..., -1]
    Tx = _centroid(q1.sum(1), xc) - _centroid(q0.sum(1), xc)
    Ty = _centroid(q1.sum(2), yc) - _centroid(q0.sum(2), yc)
    Tx = np.clip(np.nan_to_num(Tx), -800.0, 800.0)
    Ty = np.clip(np.nan_to_num(Ty), -200.0, 200.0)
    sign_a = np.tanh(np.clip(F32(E["sign_k_alpha"]) * (F32(E["sign_a0_alpha"]) - Tx), -30, 30))
    sign_b = np.tanh(np.clip(F32(E["sign_k_beta"]) * (F32(E["sign_a0_beta"]) - Ty), -30, 30))

    raw = np.stack([x_pred, y_pred, cota * sign_a, cotb * sign_b], -1)
    raw = raw / np.asarray(C["labels_scale"], F32)
    return (np.asarray(E["aff_scale"], F32) * raw + np.asarray(E["aff_bias"], F32)).astype(F32)


def error_mlp(qd, E):
    """10 raw Cholesky entries. 32 -> Dense(8, tanh) -> Dense(10, linear)."""
    xt = qd.mean(-1)                                   # time-mean -> (B,16,16)
    h = np.concatenate([xt.mean(2), xt.mean(1)], -1)   # row-avg, col-avg -> (B,32)
    h = np.tanh(h @ np.asarray(E["mlp_w1"], F32) + np.asarray(E["mlp_b1"], F32))
    return (h @ np.asarray(E["mlp_w2"], F32) + np.asarray(E["mlp_b2"], F32)).astype(F32)


# --------------------------------------------------------------- router --
def router_features(qd):
    """36 features: normalized x/y profiles (16+16) + std, skew, drift, log-charge.

    Only needed for the MoE (N>1). The N=1 arm has no router at all -- and note
    this is where the awkward-for-HLS ops live (log, sqrt, a cube, then softmax).
    """
    xp = np.maximum(qd, 0.0)
    q = xp.sum(-1)
    px, py = q.sum(2), q.sum(1)
    tot = py.sum(-1, keepdims=True) + F32(1e-9)
    pxn, pyn = px / tot, py / tot
    yc = (np.arange(16, dtype=F32) - 7.5)
    mu = (pyn * yc).sum(-1, keepdims=True)
    d = yc[None, :] - mu
    m2 = (pyn * d ** 2).sum(-1, keepdims=True)
    m3 = (pyn * d ** 3).sum(-1, keepdims=True)
    std = np.sqrt(m2 + 1.0)                            # floor of 1.0: m3/std**3 underflows
    py0, py1 = xp[..., 0].sum(1), xp[..., -1].sum(1)
    c0 = (py0 * yc).sum(-1, keepdims=True) / (py0.sum(-1, keepdims=True) + F32(1e-9))
    c1 = (py1 * yc).sum(-1, keepdims=True) / (py1.sum(-1, keepdims=True) + F32(1e-9))
    f = np.concatenate([pxn, pyn, std, m3 / std ** 3, c1 - c0, np.log(tot + 1.0)], -1)
    return np.clip(np.nan_to_num(f), -8.0, 8.0).astype(F32)


def route(qd, C):
    """Softmax gate over experts -> (B, n_experts)."""
    h = router_features(qd)
    for i, (W, b) in enumerate(zip(C["router"]["weights"], C["router"]["biases"])):
        h = h @ np.asarray(W, F32) + np.asarray(b, F32)
        if i < len(C["router"]["weights"]) - 1:
            h = np.maximum(h, 0.0)                      # relu on hidden layers only
    z = h / F32(C["router"]["temp"])
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return (e / e.sum(-1, keepdims=True)).astype(F32)


# -------------------------------------------------------------- forward --
def forward(qd, C):
    """Digitized cluster (B,16,16,2) -> the packed 14-vector (B,14)."""
    qd = np.asarray(qd, F32)
    experts = C["experts"]
    if len(experts) == 1:
        mu, chol = ansatz(qd, experts[0], C), error_mlp(qd, experts[0])
    else:
        w = route(qd, C)
        mu = sum(w[:, k:k + 1] * ansatz(qd, E, C) for k, E in enumerate(experts))
        chol = sum(w[:, k:k + 1] * error_mlp(qd, E) for k, E in enumerate(experts))
    out = np.stack([mu[:, 0], chol[:, 0], mu[:, 1], chol[:, 1],
                    mu[:, 2], chol[:, 2], mu[:, 3], chol[:, 3]], -1)
    return np.concatenate([out, chol[:, 4:]], -1).astype(F32)


# ------------------------------------------------------------ decoding --
def decode(y14, labels_scale):
    """14-vector -> physical means and marginal sigmas.

    x, y in um; cotA, cotB dimensionless (use degrees(arctan2(1, cot)) for angles).
    sigma_i = ||row i of L||, with the Cholesky diagonal passed through softplus --
    this MUST match the training loss, or the pulls come out wrong.
    """
    y14 = np.asarray(y14, F32)
    ls = np.asarray(labels_scale, F32)
    mu = y14[:, 0:8:2] * ls
    dia = 1e-9 + np.log1p(np.exp(-np.abs(y14[:, 1:8:2]))) + np.maximum(y14[:, 1:8:2], 0.0)
    off = y14[:, 8:]
    sig = np.stack([
        np.abs(dia[:, 0]),
        np.sqrt(off[:, 0] ** 2 + dia[:, 1] ** 2),
        np.sqrt(off[:, 1] ** 2 + off[:, 2] ** 2 + dia[:, 2] ** 2),
        np.sqrt(off[:, 3] ** 2 + off[:, 4] ** 2 + off[:, 5] ** 2 + dia[:, 3] ** 2),
    ], -1) * ls
    return mu, sig

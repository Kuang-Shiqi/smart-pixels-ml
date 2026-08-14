#!/usr/bin/env python
"""Train ONE arm of the symbolic-student study. Headless port of
train_digitization_ablation.ipynb / train_noise_ablation.ipynb.

This script is the CANONICAL training path. The notebooks are kept for interactive
debugging, but a notebook that is open in JupyterLab will overwrite edits made to it
on disk, which has already cost us one silently-lost fix -- so anything that must be
reproducible runs from here.

Every knob that changes the answer is a flag, and every flag lands in
OUT_DIR/summary.json. Artifacts are byte-compatible with the notebooks, so
compare_baselines_vs_symbolic.ipynb needs no changes:

    OUT_DIR/
      symbolic_n{N}_vars.parquet   predictions + truth + sigmas
      labels_scale.json            de-normalization used by the comparison notebook
      summary.json                 full provenance: quantizer, noise, metrics, CLI
      sign_init.json               calibrated TimeGrad gate (k, a0)
      expert_scalars.json          trained ansatz scalars
      history_p{1,2,3}.csv         per-epoch losses
      symbolic_n{N}.weights.h5

Physics/reference constants come from symbolic/digitize.py (arXiv:2602.15946 Sec. IV).
Never retype them here.

Examples
--------
  # paper-threshold 2-bit reference (the arm everything is compared to)
  python scripts/train_arm.py --n-bits 2 --thr-scheme paper --level-mode code

  # soft digitization: co-train the three ADC thresholds (k anneals 1 -> 67)
  python scripts/train_arm.py --n-bits 2 --trainable-thr

  # nominal noise
  python scripts/train_arm.py --n-bits 2 --noise-sigma 80

  # train clean, test noisy (robustness vs noise-aware training)
  python scripts/train_arm.py --n-bits 2 --noise-sigma 0 --test-noise-sigma 80

  # analog reference point (NOT a candidate design -- the ASIC ADC is 2-bit)
  python scripts/train_arm.py --n-bits none
"""
import argparse
import json
import math
import os
import sys
import time

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HELPERS = os.path.join(WORKDIR, "two_bit_optimization_helpers")


# ----------------------------------------------------------------- CLI --
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("data")
    g.add_argument("--dataset-dir", default="/depot/cms/users/kuang14/Smart_Pixel/"
                   "dataset_s_series/dataset_3sr/"
                   "dataset_3sr_16x16_50x12P5_centeredIncidence_parquets")
    g.add_argument("--batch", type=int, default=5000)
    g.add_argument("--timeslices", type=int, default=2)
    g.add_argument("--no-select-contained", action="store_true")
    g.add_argument("--regenerate-tfrecords", action="store_true",
                   help="rebuild the tfrecords instead of reusing them (slow)")

    g = p.add_argument_group("digitization (paper Sec. IV)")
    g.add_argument("--n-bits", default="2",
                   help="2 IS THE DEPLOYMENT POINT. 'none'/3/4 are reference only.")
    g.add_argument("--thr-scheme", default="paper",
                   choices=["paper", "lowfirst", "quantile", "linear"],
                   help="'paper' = published (248, 668, 1663) e-, 2-bit only")
    g.add_argument("--level-mode", default="code", choices=["code", "midpoint"],
                   help="'code' = bin index (Eq. 1), the only paper-comparable option")
    g.add_argument("--offset", type=float, default=None, help="default: sigma_noise = 80 e-")
    g.add_argument("--no-ste", action="store_true", help="disable the straight-through estimator")
    g.add_argument("--trainable-thr", action="store_true",
                   help="co-train the thresholds; forces k_init=1 with cosine anneal to k_max")
    g.add_argument("--t-min", type=float, default=240.0,
                   help="Eq. 3 floor; the learned T0 cannot fall below this")
    g.add_argument("--parametrization", default="paper", choices=["paper", "legacy"])
    g.add_argument("--k-init", type=float, default=None)
    g.add_argument("--k-max", type=float, default=None)

    g = p.add_argument_group("noise (Eq. 6, on the ANALOG charge, digitized after)")
    g.add_argument("--noise-mu", type=float, default=0.0)
    g.add_argument("--noise-sigma", type=float, default=0.0,
                   help="e-. 0 = clean. Nominal = 80 (Cadence Virtuoso / Spectre)")
    g.add_argument("--test-noise-sigma", type=float, default=None,
                   help="evaluate at a DIFFERENT sigma than training (mismatch arm)")

    g = p.add_argument_group("model")
    g.add_argument("--variant", default="barycenter")
    g.add_argument("--n-experts", type=int, default=1)
    g.add_argument("--router-hidden", default=None,
                   help="comma-separated, e.g. '32,16'. Default: by n_experts")
    g.add_argument("--router-temp", type=float, default=1.0)

    g = p.add_argument_group("schedule")
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--epochs-p1", type=int, default=150)
    g.add_argument("--epochs-p2", type=int, default=200)
    g.add_argument("--epochs-p3", type=int, default=120)
    g.add_argument("--seed", type=int, default=42)

    g = p.add_argument_group("output")
    g.add_argument("--out-root", default="/depot/cms/private/users/kuang14/Smart_Pixel")
    g.add_argument("--out-dir", default=None, help="override the auto-generated run directory")
    g.add_argument("--tag-suffix", default="", help="appended to the auto tag")
    g.add_argument("--label-scale-mode", default="train", choices=["train", "align"],
                   help="'train' (default) reproduces every run made so far: the ansatz and "
                        "the parquet both use the TRAIN split's labels_scale. 'align' rescales "
                        "predictions into the EVAL split's normalization before dumping. The "
                        "two splits carry different 99th-percentile scales (0.42%% on x, 0.64%% "
                        "on cotB), so 'train' leaves a slope of that size in the residuals. "
                        "Do not mix modes across runs you intend to compare.")
    g.add_argument("--overwrite", action="store_true",
                   help="allow writing into a run directory that already has a parquet")
    g.add_argument("--dry-run", action="store_true",
                   help="resolve config, print the plan and exit without touching the GPU")

    a = p.parse_args(argv)
    a.n_bits = None if str(a.n_bits).lower() in ("none", "analog", "null") else int(a.n_bits)
    a.select_contained = not a.no_select_contained
    a.ste = not a.no_ste
    if a.router_hidden is not None:
        a.router_hidden = tuple(int(v) for v in a.router_hidden.split(",") if v.strip())
    else:
        a.router_hidden = {4: (32, 16), 2: (16,)}.get(a.n_experts, ())
    # k is annealed only when the thresholds need gradient: a large fixed k makes the
    # sigmoids step-like, which is right for a frozen quantizer and useless for a
    # trainable one (DigitizeLayer raises on the bad combination).
    dflt_k = (1.0, 67.0) if a.trainable_thr else (50.0, 50.0)
    a.k_init = dflt_k[0] if a.k_init is None else a.k_init
    a.k_max = dflt_k[1] if a.k_max is None else a.k_max

    if a.thr_scheme == "paper" and a.n_bits not in (2, None):
        p.error("--thr-scheme paper publishes two-bit thresholds only; use 'lowfirst' for 3/4-bit")
    if a.trainable_thr and a.n_bits is None:
        p.error("--trainable-thr with --n-bits none: nothing to train in analog mode")
    return a


def run_tag(a):
    bits = "analog" if a.n_bits is None else f"{a.n_bits}bit"
    tag = bits if a.n_bits is None else f"{bits}_{a.thr_scheme}_{a.level_mode}"
    if a.trainable_thr:
        tag += "_trainthr"
    if a.noise_sigma > 0:
        tag += f"_noise{int(a.noise_sigma)}"
    if a.test_noise_sigma is not None:
        tag += f"_test{int(a.test_noise_sigma)}"
    return tag + a.tag_suffix


# ------------------------------------------------------------- helpers --
def i68(r):
    """paper metric: half-width of the minimal interval containing 68% of residuals"""
    import numpy as np
    r = np.sort(np.asarray(r)[np.isfinite(r)])
    n = len(r); k = int(np.ceil(0.68 * n))
    return (r[-1] - r[0]) / 2.0 if k >= n else (r[k:] - r[:n - k]).min() / 2.0


def pull_fit(p):
    import numpy as np
    from scipy.optimize import curve_fit
    gauss = lambda x, A, mu, s: A * np.exp(-(x - mu) ** 2 / (2 * s ** 2))
    h = np.histogram(p, bins=np.linspace(-5, 5, 100))
    xd = h[1][:-1] + np.diff(h[1]) / 2
    try:
        pars, _ = curve_fit(gauss, xd, h[0], p0=[h[0].max(), 0, 1], maxfev=5000)
        return float(pars[1]), float(abs(pars[2]))
    except Exception:
        return float("nan"), float("nan")


def freeze_adc_thresholds(q, model):
    """Freeze the learned ADC thresholds, whatever the Keras build.

    `keras.Variable.trainable` is READ-ONLY in some Keras 3 builds -- assigning it
    raises "AttributeError: can't set attribute" -- while `.assign()` on the same
    variable works fine. Try the variable, then the sublayer, then the private flag,
    and verify against `model.trainable_variables`: that list is the only thing that
    decides whether the optimizer still moves the thresholds.

    Freezing the sublayer is equivalent here, because threshold_deltas_raw is the
    SoftQuantizeLayer's only trainable weight (levels and k are both non-trainable).
    """
    v = q.threshold_deltas_raw
    live = lambda: any(w is v for w in model.trainable_variables)
    for how, apply in (("variable", lambda: setattr(v, "trainable", False)),
                       ("sublayer", lambda: setattr(q, "trainable", False)),
                       ("private", lambda: setattr(v, "_trainable", False))):
        try:
            apply()
        except AttributeError:
            continue
        if not live():
            try:                       # keep `.trainable` READING False as well
                v._trainable = False
            except Exception:
                pass
            return how
    raise RuntimeError(
        "ADC thresholds are STILL in model.trainable_variables -- they would keep moving "
        "while phases 2-3 fit the physics and the sigmas against a shifting input.")


# ---------------------------------------------------------------- main --
def main(argv=None):
    a = parse_args(argv)
    tag = run_tag(a)
    out_dir = a.out_dir or os.path.join(a.out_root, f"digi_{tag}_nexp{a.n_experts}")

    print("=" * 78)
    print(f"ARM  : {tag}  (N={a.n_experts})")
    print(f"OUT  : {out_dir}")
    print(f"CMD  : {' '.join(sys.argv)}")
    print("=" * 78)
    if a.dry_run:
        print(json.dumps({k: v for k, v in vars(a).items()}, indent=1, default=str))
        return 0

    pq_path = os.path.join(out_dir, f"symbolic_n{a.n_experts}_vars.parquet")
    if os.path.exists(pq_path) and not a.overwrite:
        print(f"REFUSING: {pq_path} already exists. Pass --overwrite, or --tag-suffix to "
              f"branch off a new run directory. Never silently replace a prior parquet.")
        return 2
    os.makedirs(out_dir, exist_ok=True)

    os.chdir(WORKDIR)
    sys.path.insert(0, HELPERS)
    assert "TF_USE_LEGACY_KERAS" not in os.environ, \
        "TF_USE_LEGACY_KERAS breaks training (numerical incompatibility with Keras 3)"

    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import tensorflow_probability as tfp
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except Exception:
            pass

    from prepare_tfrecords import generate_tfrecords, load_tfrecords
    from loss import custom_loss
    from models.student_max import build_student_max, pack_14
    from symbolic.digitize import (build_thresholds, levels_for, DigitizeLayer, AnnealK,
                                   SIGMA_NOISE_E, FIVE_SIGMA_E, PAPER_THRESHOLDS_2BIT, PAPER_REF)

    offset = SIGMA_NOISE_E if a.offset is None else a.offset
    tf.random.set_seed(a.seed); np.random.seed(a.seed)
    print(f"TF {tf.__version__} | GPUs: {tf.config.list_physical_devices('GPU')}")

    # ---- 1. data ---------------------------------------------------------
    noise = -1 if a.noise_sigma <= 0 else [float(a.noise_mu), float(a.noise_sigma)]
    _, _, tfr_tr, tfr_val = generate_tfrecords(
        dataset_dir=a.dataset_dir, model_type="ViT_Max",   # tfrecord format only; NO teacher
        train_batch_size=a.batch, val_batch_size=a.batch,
        select_contained=a.select_contained, timeslices=a.timeslices,
        tfrecords_exist=not a.regenerate_tfrecords, seed=a.seed)
    tg, vg = load_tfrecords(tfr_tr, tfr_val, noise=noise, digitize=False, seed=a.seed)

    if a.test_noise_sigma is None:
        eval_gen, eval_noise = vg, a.noise_sigma
    else:
        tn = -1 if a.test_noise_sigma <= 0 else [float(a.noise_mu), float(a.test_noise_sigma)]
        _, eval_gen = load_tfrecords(tfr_tr, tfr_val, noise=tn, digitize=False, seed=a.seed)
        eval_noise = a.test_noise_sigma
        print(f"cross-arm: trained at sigma={a.noise_sigma} e-, evaluated at sigma={eval_noise} e-")

    scale_tr = json.load(open(os.path.join(tfr_tr, "metadata.json")))["labels_scale"]
    scale_ev = json.load(open(os.path.join(tfr_val, "metadata.json")))["labels_scale"]
    labels_scale = list(scale_tr)                    # what the ansatz is trained against
    ratio = np.asarray(scale_tr, float) / np.asarray(scale_ev, float)
    print(f"labels_scale train: {[round(v, 6) for v in scale_tr]}")
    print(f"labels_scale eval : {[round(v, 6) for v in scale_ev]}")
    if not np.allclose(ratio, 1.0, atol=1e-6):
        print(f"  NOTE: the two splits carry DIFFERENT scales, ratio-1 = "
              f"{np.round(100 * (ratio - 1), 3)} %. The model is fit against the train "
              f"normalization and evaluated against the eval one, so that percentage is a "
              f"slope in the residuals. label_scale_mode={a.label_scale_mode!r}.")
    print(f"train batches: {len(tg)} | val batches: {len(vg)}")

    xb, yb = np.asarray(tg[0][0]), np.asarray(tg[0][1])
    assert xb.shape[1:] == (16, 16, a.timeslices) and yb.shape[1] == 4
    print(f"x: {xb.shape} | y: {yb.shape} | charge range: {float(xb.min())} .. {float(xb.max())}")

    # ---- 2. ADC thresholds ----------------------------------------------
    if a.n_bits is None:
        thresholds = levels = bin_edges = None
        thr_prov = dict(n_bits=None, source="analog passthrough -- no quantizer")
        print("analog inputs -- DigitizeLayer is a passthrough")
    else:
        Xthr = np.concatenate([np.asarray(tg[i][0], "float32") for i in range(min(4, len(tg)))])
        thresholds, bin_edges, thr_prov = build_thresholds(
            Xthr, a.n_bits, scheme=a.thr_scheme, offset=offset)
        levels = levels_for(a.n_bits, a.level_mode, bin_edges)
        occ = np.bincount(np.digitize(Xthr[Xthr > offset], thresholds),
                          minlength=2 ** a.n_bits) / max((Xthr > offset).sum(), 1)
        thr_prov["occupancy_per_code"] = [float(v) for v in occ]
        print(f"thresholds (e-): {np.round(thresholds, 1)}")
        print(f"  source       : {thr_prov['source']}")
        print(f"  T0 = {thresholds[0]:.1f} e- = {thr_prov['t0_over_sigma_noise']:.2f} sigma_noise"
              f"  (5 sigma = {FIVE_SIGMA_E:.0f} e-, below it: {thr_prov['t0_below_five_sigma']})")
        if thr_prov.get("published_thresholds_e"):
            print(f"  vs published : "
                  f"{np.round(np.asarray(thresholds) - np.asarray(thr_prov['published_thresholds_e']), 1)} e-")
        for n in thr_prov.get("notes", []):
            print(f"  NOTE: {n}")
        print(f"levels ({a.level_mode}): {np.round(levels, 2)}")
        print(f"occupancy/code: {np.round(occ, 3)}")
        del Xthr

    def digitize_np(x):
        """numpy mirror of DigitizeLayer's hard path, for the sign calibration."""
        x = np.asarray(x, "float32")
        if a.n_bits is None:
            return x
        return levels[np.clip(np.digitize(x, thresholds), 0, len(levels) - 1)].astype("float32")

    # ---- 3. what the noise does to the input (measured, not assumed) -----
    # NOTE: this dataset carries INTRINSIC negative charge with the noise off -- ~2.9% of
    # pixels, and the minimum runs to about -3400 e- (shaping undershoot in the convolved
    # simulation). So "negative implies noise" is false here, and an RMS over the negative
    # side is dominated by that rare tail. Use the MEDIAN of |negative|, which for a
    # half-normal is 0.6745*sigma: once the noise is on it makes ~half of all pixels
    # negative, so the median sits in the noise bulk and ignores the intrinsic tail.
    Xn = np.concatenate([np.asarray(tg[i][0], "float32") for i in range(min(4, len(tg)))])
    neg = np.abs(Xn[Xn < 0.0])
    sig_est = float(np.median(neg) / 0.674489750196) if neg.size else 0.0
    frac_neg = float(neg.size / Xn.size)
    print(f"\ngenerator noise: requested sigma = {a.noise_sigma:.0f} e- | robust estimate from "
          f"the negative side = {sig_est:.0f} e- ({frac_neg:.1%} of pixels < 0)")
    noise_diag = dict(sigma_requested_e=a.noise_sigma, sigma_measured_e=sig_est,
                      frac_pixels_negative=frac_neg,
                      estimator="median(|q|, q<0)/0.6745, robust to the dataset's "
                                "intrinsic negative-charge tail")
    if a.noise_sigma > 0:
        assert 0.6 * a.noise_sigma < sig_est < 1.4 * a.noise_sigma, (
            f"generator noise sigma looks like {sig_est:.0f} e-, not {a.noise_sigma:.0f} -- "
            "check the NOISE plumbing before training")
        assert frac_neg > 0.2, (f"only {frac_neg:.1%} of pixels are negative; with sigma="
                                f"{a.noise_sigma:.0f} e- the noise should push roughly half "
                                "of the empty pixels below zero -- noise may not be applied")
    else:
        print(f"  (noise OFF -- the {frac_neg:.1%} negative pixels are intrinsic to the "
              f"dataset, not injected)")
    del Xn

    if a.n_bits is not None:
        tg_clean, _ = load_tfrecords(tfr_tr, tfr_val, noise=-1, digitize=False, seed=a.seed)
        Xq = np.concatenate([np.asarray(tg_clean[i][0], "float32")
                             for i in range(min(4, len(tg_clean)))])
        rng = np.random.default_rng(a.seed)
        Xr = Xq + rng.normal(a.noise_mu, max(a.noise_sigma, 0.0), Xq.shape).astype("float32")
        code = lambda z: np.digitize(z, thresholds)
        empty = Xq <= 0.0
        rate = float(((code(Xr) > 0) & empty).sum() / max(empty.sum(), 1))
        analytic = (0.0 if a.noise_sigma <= 0 else
                    0.5 * math.erfc((thresholds[0] - a.noise_mu) / (a.noise_sigma * np.sqrt(2.0))))
        flip = code(Xr) != code(Xq)
        sig_px = Xq > thresholds[0]
        noise_diag.update(
            t0_over_sigma=float(thresholds[0] / max(a.noise_sigma, 1e-9)),
            pedestal_fire_rate=rate, pedestal_fire_rate_analytic=float(analytic),
            hot_pixels_per_frame=float(rate * empty.mean() * 16 * 16 * a.timeslices),
            code_flip_all=float(flip.mean()), code_flip_signal=float(flip[sig_px].mean()))
        t0_sig = (f"{noise_diag['t0_over_sigma']:.2f} sigma_noise" if a.noise_sigma > 0
                  else "n/a (noise off)")
        print(f"T0 = {thresholds[0]:.0f} e- = {t0_sig}")
        print(f"  pedestal pixels firing : {rate:.2e} measured | {analytic:.2e} analytic")
        print(f"  hot pixels per frame   : {noise_diag['hot_pixels_per_frame']:.3f}")
        print(f"  code flips all/signal  : {flip.mean():.3%} / {flip[sig_px].mean():.3%}")
        if rate > 0.01:
            print("  WARNING: >1% of empty pixels fire -- clusters read wider than they are "
                  "and the centroid drift the sign gate needs is buried. Raise T0.")
        del Xq, Xr, tg_clean

    # ---- 4. calibrate the TimeGrad sign gates from TRAIN data ------------
    # 1-feature logistic P(sign=+) = sigma(w*T + b) maps to tanh(k*(a0 - T)), k=-w/2, a0=-b/w.
    # alpha <- Tx (cols @ 50 um); beta <- Ty (rows @ 12.5 um). v3 axes, uncrossed.
    def drifts_np(X):
        q = np.maximum(np.asarray(X, "float32"), 0.0)
        row, col = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
        xw, yw = col * 50.0, row * 12.5
        q0 = q[..., 0].sum((1, 2)) + 1e-6
        q1 = q[..., 1].sum((1, 2)) + 1e-6
        Tx = (q[..., 1] * xw).sum((1, 2)) / q1 - (q[..., 0] * xw).sum((1, 2)) / q0
        Ty = (q[..., 1] * yw).sum((1, 2)) / q1 - (q[..., 0] * yw).sum((1, 2)) / q0
        return Tx, Ty

    def collect(gen, n_batches=None):
        n = len(gen) if n_batches is None else min(n_batches, len(gen))
        Xs, Ys = zip(*[(np.asarray(gen[i][0]), np.asarray(gen[i][1])) for i in range(n)])
        return np.concatenate(Xs), np.concatenate(Ys)

    Xtr_, Ytr_ = collect(tg, n_batches=8)
    Xva_, Yva_ = collect(vg)
    Tx_tr, Ty_tr = drifts_np(digitize_np(Xtr_))
    Tx_va, Ty_va = drifts_np(digitize_np(Xva_))

    sign_init, sign_cal = {}, {}
    for nm, T_tr, T_va, i in [("alpha", Tx_tr, Tx_va, 2), ("beta", Ty_tr, Ty_va, 3)]:
        clf = LogisticRegression(max_iter=5000).fit(T_tr.reshape(-1, 1),
                                                    (Ytr_[:, i] > 0).astype(int))
        w, b = float(clf.coef_[0, 0]), float(clf.intercept_[0])
        k, a0 = -w / 2.0, -b / w
        # Use the gate EXACTLY as the ansatz evaluates it: sign(tanh(k*(a0-T))) = sign(k*(a0-T)).
        # The notebook dropped k here, so a fit that returned k < 0 reported a healthy accuracy
        # while the model ran the opposite sign -- caught only later by the init-agreement check.
        hard = np.where(k * (a0 - T_va) > 0, 1.0, -1.0)
        agr = float((hard == np.sign(Yva_[:, i])).mean())
        bal = float(balanced_accuracy_score((Yva_[:, i] > 0).astype(int), (hard > 0).astype(int)))
        sign_init[f"initial_sign_k_{nm}"] = k
        sign_init[f"initial_sign_a0_{nm}"] = a0
        sign_cal[nm] = dict(k=k, a0=a0, val_agreement=agr, val_balanced_acc=bal,
                            k_sign_negative=bool(k < 0))
        print(f"cot{nm}: k={k:+.4f}/um  a0={a0:+.3f} um | VAL agree {agr:.4f}  bal-acc {bal:.4f}"
              + ("   [k<0: convention flip, allowed]" if k < 0 else ""))
        assert bal > 0.70, (f"cot{nm} sign calibration {bal:.4f} at {tag} -- the gate is not "
                            "learning the sign at all. Check thresholds/level_mode.")
    json.dump(sign_init, open(os.path.join(out_dir, "sign_init.json"), "w"), indent=1)
    del Xtr_, Ytr_

    # ---- 5. model --------------------------------------------------------
    def make_digitizer():
        return DigitizeLayer(n_bits=a.n_bits, thresholds=thresholds, levels=levels,
                             threshold_offset=(a.t_min if a.trainable_thr else offset),
                             ste=a.ste, trainable_thresholds=a.trainable_thr,
                             trainable_levels=False, initial_k=a.k_init,
                             parametrization=a.parametrization, name="digitize")

    class RouterFeatures(tf.keras.layers.Layer):
        # HARDENED: relu charge, std floored, finite-guard + clip.
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

    # No AxisFixExpert wrapper: the transpose it used to patch is fixed inside ansatz.py
    # (v3), for the pitches, widths and sign observables as well as the position slots.
    # Re-adding a wrapper would double-swap and reproduce the original bug exactly.
    def make_expert():
        return build_student_max(a.variant,
                                 ansatz_kwargs={"labels_scale": labels_scale, **sign_init})

    class MoECore(tf.keras.Model):
        def __init__(self, n_experts, router_hidden, temp, **kw):
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
            xd = self.digi(x, training=training)   # digitize ONCE, share with router+experts
            w = self._route_d(xd)
            outs = [e(xd, training=training) for e in self.experts]
            means = tf.add_n([w[:, k:k + 1] * outs[k][0] for k in range(self.n_experts)])
            chol = tf.add_n([w[:, k:k + 1] * outs[k][1] for k in range(self.n_experts)])
            return means, chol

    class SingleCore(tf.keras.Model):
        def __init__(self, **kw):
            super().__init__(**kw)
            self.digi = make_digitizer()
            self.experts = [make_expert()]

        def call(self, x, training=False):
            return self.experts[0](self.digi(x, training=training), training=training)

    class PackedStudent(tf.keras.Model):
        def __init__(self, core, **kw):
            super().__init__(**kw)
            self.core = core

        def call(self, x, training=False):
            mu, chol = self.core(x, training=training)
            return pack_14(mu, chol)

    core = (SingleCore(name="single_core") if a.n_experts == 1
            else MoECore(a.n_experts, a.router_hidden, a.router_temp, name="moe_core"))
    model = PackedStudent(core, name=f"symbolic_n{a.n_experts}")
    assert model(xb[:64]).shape[-1] == 14
    n_router = core.router.count_params() if a.n_experts > 1 else 0
    print(f"\ntotal params: {model.count_params()} | router: {n_router} "
          f"| per-expert: {core.experts[0].count_params()}")

    names = [v.name for v in model.trainable_variables]
    for need in ("sign_k_alpha", "sign_a0_alpha", "sign_k_beta", "sign_a0_beta"):
        found = sum(need in n for n in names)
        assert found == a.n_experts, f"{need}: found {found}, expected {a.n_experts} -- stale ansatz.py"

    # axis convention: probe the LOADED class. One pixel at (row 0, col 15): x is the
    # column axis at 50 um, so the x slot must read (15-7.5)*50 = +375 um.
    from symbolic import PhysicsAnsatz as _PA
    _z = np.zeros((1, 16, 16, a.timeslices), "float32"); _z[0, 0, 15, :] = 1000.0
    _o = np.asarray(_PA(variant="barycenter", labels_scale=[1.0] * 4)(tf.constant(_z)))[0]
    print(f"axis probe: x-slot={_o[0]:+.1f} um  y-slot={_o[1]:+.1f} um  (expect x=+375.0)")
    assert _o[0] > 300.0, f"AXIS FIX NOT LIVE: x-slot={_o[0]:+.1f} -- stale ansatz.py"

    probe_in = np.asarray(xb[:256], "float32")
    probe_out = np.asarray(core.digi(tf.constant(probe_in), training=False))
    if a.n_bits is None:
        assert np.allclose(probe_in, probe_out), "analog mode must be a passthrough"
        print("digitizer: passthrough (analog)")
    else:
        u = np.unique(probe_out)
        assert len(u) <= 2 ** a.n_bits, f"{len(u)} distinct outputs, expected <= {2 ** a.n_bits}"
        print(f"digitizer: {a.n_bits}-bit live | distinct outputs {len(u)}: {np.round(u, 2)}")

    # ---- 6. pre-train checks --------------------------------------------
    cb, lb = np.asarray(vg[0][0]), np.asarray(vg[0][1])
    mu0 = np.asarray(core(cb, training=False)[0])
    for nm, i in [("x", 0), ("y", 1)]:
        cx = np.corrcoef(mu0[:, i], lb[:, 0])[0, 1]
        cy = np.corrcoef(mu0[:, i], lb[:, 1])[0, 1]
        print(f"  {nm}-slot: corr(true_x)={cx:+.3f}  corr(true_y)={cy:+.3f}")
        assert abs([cx, cy][i]) > 0.8, f"{nm} not on its own axis -- axis fix broken, STOP"
    for nm, i in [("cotA", 2), ("cotB", 3)]:
        m = mu0[:, i] != 0
        agr = float((np.sign(mu0[m, i]) == np.sign(lb[m, i])).mean())
        ceiling = sign_cal["alpha" if nm == "cotA" else "beta"]["val_balanced_acc"]
        print(f"  {nm}: init sign agreement {agr:.4f} (gate ceiling {ceiling:.4f}, "
              f"on {m.mean():.1%} nonzero-|cot| events)")
        # Bar is tied to the MEASURED gate ceiling for this arm, not a constant: under
        # noise the drift is smeared and the ceiling itself drops, so a fixed 0.90 would
        # reject a model that is doing as well as the observable allows.
        assert agr > min(0.90, ceiling - 0.05), (
            f"{nm} init sign {agr:.4f} vs gate ceiling {ceiling:.4f} -- the model is not "
            "reproducing its own calibrated gate. Do NOT train.")
    l0 = float(tf.reduce_mean(custom_loss(tf.constant(yb, tf.float32), model(xb, training=False))))
    assert np.isfinite(l0), "init loss is not finite"
    bad = sum(0 if np.isfinite(model(vg[i][0], training=False).numpy()).all() else 1
              for i in range(len(vg)))
    assert bad == 0, f"{bad} val batches produce NaN at init"
    print(f"  init custom_loss: {l0:.4f} | NaN scan: 0/{len(vg)} bad")
    print("all pre-train checks passed")

    # ---- 7. training -----------------------------------------------------
    class NaNStop(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, b, logs=None):
            v = (logs or {}).get("loss")
            if v is not None and not np.isfinite(v):
                print(f"\nNaN loss at batch {b} -- stopping")
                self.model.stop_training = True

    anneal_cb = (AnnealK(core.digi, k_init=a.k_init, k_max=a.k_max, verbose=1)
                 if (a.n_bits is not None and a.trainable_thr) else None)
    cbs = lambda name, patience: [
        tf.keras.callbacks.CSVLogger(os.path.join(out_dir, f"history_{name}.csv")),
        NaNStop(),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                         restore_best_weights=True, verbose=1)]

    def mse_means(y_true, y_pred14):
        mu = tf.gather(y_pred14, [0, 2, 4, 6], axis=-1)
        return tf.reduce_mean(tf.square(y_true - mu), axis=-1)

    # phase 1: means only. Starting on NLL lets sigma-inflation kill the mean gradients.
    print("\n" + "-" * 30 + " phase 1: means (MSE) " + "-" * 30)
    model.compile(optimizer=tf.keras.optimizers.Adam(a.lr, clipnorm=1.0), loss=mse_means)
    t0 = time.time()
    h1 = model.fit(tg, validation_data=vg, epochs=a.epochs_p1, shuffle=False, verbose=2,
                   callbacks=([anneal_cb] if anneal_cb else []) + cbs("p1", 30))
    print(f"phase 1 done in {time.time() - t0:.0f}s")

    # Thresholds are learned HERE and nowhere else: phase 1 is the only stretch where k is
    # soft enough for them to receive gradient. Freeze at the end so phases 2-3 fit against
    # a fixed ADC -- exactly the quantizer the chip would ship.
    thresholds_learned = None
    if a.n_bits is not None:
        print(f"  thresholds after phase 1 (e-): {np.round(np.asarray(core.digi.thresholds), 1)}")
        if a.trainable_thr:
            core.digi.quantizer.log_k.assign([np.log(np.float32(a.k_max))])
            how = freeze_adc_thresholds(core.digi.quantizer, model)
            thresholds_learned = [float(v) for v in np.asarray(core.digi.thresholds)]
            print(f"  ADC frozen (via {how}). learned T (e-): {np.round(thresholds_learned, 1)}")
            print(f"  moved from init (e-)      : "
                  f"{np.round(np.asarray(thresholds_learned) - np.asarray(thresholds), 1)}")
            if a.n_bits == 2:
                print(f"  vs published (e-)         : "
                      f"{np.round(np.asarray(thresholds_learned) - np.asarray(PAPER_THRESHOLDS_2BIT), 1)}")
            assert thresholds_learned[0] >= a.t_min - 1e-3, "T0 fell below T_min"

    mu_v = np.asarray(core(np.asarray(vg[0][0]), training=False)[0])
    yv = np.asarray(vg[0][1])
    for nm, i in [("x", 0), ("y", 1)]:
        r = np.corrcoef(mu_v[:, i], yv[:, i])[0, 1]
        print(f"phase-1 {nm}: corr(pred, true) = {r:.4f}")
        assert r > 0.5, f"{nm} means still dead after phase 1 -- STOP"

    print("\n" + "-" * 30 + " phase 2: means+cov (NLL) " + "-" * 30)
    model.compile(optimizer=tf.keras.optimizers.Adam(a.lr / 3, clipnorm=1.0), loss=custom_loss)
    t0 = time.time()
    h2 = model.fit(tg, validation_data=vg, epochs=a.epochs_p2, shuffle=False, verbose=2,
                   callbacks=cbs("p2", 50))
    print(f"phase 2 done in {time.time() - t0:.0f}s")

    # phase 3: error head only, under a corrected NLL. loss.custom_loss uses K.sum plus a
    # relu diagonal and a clip floor that zeros gradients on floored events, so sigma never
    # trains. Here the means are frozen and only the error MLP moves, under mean(-log_prob)
    # with a SOFTPLUS diagonal. The parquet dump below uses the SAME decode, so pulls land ~1.
    def custom_loss_fixed(y, p):
        mu = p[:, 0:8:2]
        Mdia = 1e-9 + tf.nn.softplus(p[:, 1:8:2])
        Mcov = p[:, 8:]
        z = tf.zeros_like(Mdia[:, 0])
        L = tf.transpose(tf.stack([
            tf.stack([Mdia[:, 0], z, z, z]),
            tf.stack([Mcov[:, 0], Mdia[:, 1], z, z]),
            tf.stack([Mcov[:, 1], Mcov[:, 2], Mdia[:, 2], z]),
            tf.stack([Mcov[:, 3], Mcov[:, 4], Mcov[:, 5], Mdia[:, 3]]),
        ]), perm=[2, 0, 1])
        return tf.reduce_mean(-tfp.distributions.MultivariateNormalTriL(loc=mu, scale_tril=L).log_prob(y))

    print("\n" + "-" * 30 + " phase 3: error head (corrected NLL) " + "-" * 20)
    for e in core.experts:
        e.get_layer("physics_ansatz").trainable = False
    if a.n_bits is not None and a.trainable_thr:
        tv = core.digi.quantizer.threshold_deltas_raw
        assert not any(w is tv for w in model.trainable_variables), \
            "ADC thresholds still in model.trainable_variables -- freeze failed in phase 1"
        print(f"ADC frozen, thresholds (e-): {np.round(np.asarray(core.digi.thresholds), 1)}")
    print("trainable now:", [v.name for v in model.trainable_variables])
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4, clipnorm=1.0), loss=custom_loss_fixed)
    t0 = time.time()
    h3 = model.fit(tg, validation_data=vg, epochs=a.epochs_p3, shuffle=False, verbose=2,
                   callbacks=cbs("p3", 25))
    print(f"phase 3 done in {time.time() - t0:.0f}s (val_loss should be NEGATIVE)")
    wpath = os.path.join(out_dir, f"symbolic_n{a.n_experts}.weights.h5")
    model.save_weights(wpath)
    print("saved weights ->", wpath)

    # ---- 8. evaluate + dump ---------------------------------------------
    softplus = lambda z: np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)
    P, S14, Yt = [], [], []
    for i in range(len(eval_gen)):
        cbx, lby = eval_gen[i]
        s14 = model(cbx, training=False).numpy()
        P.append(s14[:, [0, 2, 4, 6]]); S14.append(s14); Yt.append(np.asarray(lby))
    P, S14, Yt = np.concatenate(P), np.concatenate(S14), np.concatenate(Yt)

    # The ansatz is fit against the TRAIN normalization; the eval labels carry the EVAL
    # one. 'align' converts predictions into the eval normalization so a single scale
    # de-normalizes both columns correctly. 'train' reproduces every earlier run.
    if a.label_scale_mode == "align":
        P = P * ratio[None, :]
        S14[:, [0, 2, 4, 6]] *= ratio[None, :]
        dump_scale = list(scale_ev)
    else:
        dump_scale = list(scale_tr)

    Mdia = 1e-9 + softplus(S14[:, 1:8:2])
    Mcov = S14[:, 8:]
    sig = np.stack([
        np.abs(Mdia[:, 0]),
        np.sqrt(Mcov[:, 0] ** 2 + Mdia[:, 1] ** 2),
        np.sqrt(Mcov[:, 1] ** 2 + Mcov[:, 2] ** 2 + Mdia[:, 2] ** 2),
        np.sqrt(Mcov[:, 3] ** 2 + Mcov[:, 4] ** 2 + Mcov[:, 5] ** 2 + Mdia[:, 3] ** 2),
    ], axis=1)
    if a.label_scale_mode == "align":
        sig = sig * ratio[None, :]

    df = pd.DataFrame(P, columns=["x", "y", "cotA", "cotB"])
    for i, c in enumerate(["xtrue", "ytrue", "cotAtrue", "cotBtrue"]):
        df[c] = Yt[:, i]
    for i, c in enumerate(["sigmax", "sigmay", "sigmacotA", "sigmacotB"]):
        df[c] = sig[:, i]
    df.to_parquet(pq_path)
    json.dump({"labels_scale": dump_scale}, open(os.path.join(out_dir, "labels_scale.json"), "w"))

    ls = np.asarray(dump_scale, float)
    inverse_cot = lambda c: np.arctan2(1.0, np.asarray(c, float))
    metrics = {}
    print(f"\nevaluated {len(P)} events at test sigma_noise = {eval_noise} e-")
    print(f"{'var':<6}{'unit':>5}{'mean':>10}{'I68':>10}{'pull mu':>10}{'pull sig':>10}")
    for nm, i, unit in [("x", 0, "um"), ("y", 1, "um"), ("cotA", 2, "deg"), ("cotB", 3, "deg")]:
        if unit == "um":
            t, p_, sg = Yt[:, i] * ls[i], P[:, i] * ls[i], sig[:, i] * ls[i]
        else:
            t = np.degrees(inverse_cot(Yt[:, i] * ls[i]))
            p_ = np.degrees(inverse_cot(P[:, i] * ls[i]))
            pu = np.degrees(inverse_cot((P[:, i] + sig[:, i]) * ls[i]))
            pd_ = np.degrees(inverse_cot((P[:, i] - sig[:, i]) * ls[i]))
            sg = 0.5 * (np.abs(pu - p_) + np.abs(pd_ - p_))
        r = t - p_
        pm, ps = pull_fit(r / (sg + 1e-12))
        metrics[nm] = dict(unit=unit, mean=float(r.mean()), std=float(r.std()),
                           I68=float(i68(r)), pull_mu=pm, pull_sigma=ps)
        print(f"{nm:<6}{unit:>5}{r.mean():10.4f}{i68(r):10.4f}{pm:10.3f}{ps:10.3f}")
    for nm, i in [("cotA", 2), ("cotB", 3)]:
        ti, pr = np.sign(Yt[:, i]), np.sign(P[:, i])
        conf = np.abs(Yt[:, i]) > np.quantile(np.abs(Yt[:, i]), 0.5)
        metrics[nm]["sign_acc"] = float((pr == ti).mean())
        metrics[nm]["sign_acc_confident_half"] = float((pr[conf] == ti[conf]).mean())
        print(f"{nm} sign accuracy: {metrics[nm]['sign_acc']:.4f} "
              f"(confident half: {metrics[nm]['sign_acc_confident_half']:.4f})")

    # ---- 9. expert scalars + router usage --------------------------------
    expert_scalars, usage, ent = {}, None, None
    for k, e in enumerate(core.experts):
        ans = getattr(e, "inner", e).get_layer("physics_ansatz")
        expert_scalars[f"expert_{k}"] = {w.name: np.asarray(w.numpy()).tolist() for w in ans.weights}
    json.dump(expert_scalars, open(os.path.join(out_dir, "expert_scalars.json"), "w"),
              indent=1, default=float)
    if a.n_experts > 1:
        Wv = np.concatenate([core.route(vg[i][0]).numpy() for i in range(len(vg))])
        usage = Wv.mean(0)
        ent = float(-(Wv * np.log(Wv + 1e-12)).sum(1).mean())
        print(f"router usage: {usage.round(3)} | mean entropy {ent:.3f} (max {np.log(a.n_experts):.3f})")

    # ---- 10. summary -----------------------------------------------------
    summary = {
        "script": os.path.relpath(os.path.abspath(__file__), WORKDIR),
        "cli": sys.argv,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(a).items()},
        "training": "pure regression (NO teacher/KD) | phase1 MSE means -> phase2 orig-NLL "
                    "-> phase3 corrected-NLL error head",
        "sign_source": "in-ansatz TimeGrad: sign=tanh(k*(a0-drift)); alpha<-Tx, beta<-Ty "
                       "(uncrossed, v3 axes)",
        "axis_fix": "in-ansatz v3: cols=x@50um, rows=y@12.5um (measured)",
        "sigma_decode": "softplus diagonal, sigma_i = ||row_i(L)||, Sigma = L L^T",
        "sign_init": sign_init,
        "sign_calibration": sign_cal,
        "variant": a.variant, "n_experts": a.n_experts,
        "digitization": (core.digi.describe(provenance=thr_prov) if a.n_bits is not None
                         else dict(n_bits=None, mode="analog passthrough")),
        "level_mode_requested": (None if a.n_bits is None else a.level_mode),
        "level_mode_is_paper_equivalent": (None if a.n_bits is None else a.level_mode == "code"),
        "k_schedule": (dict(k_init=a.k_init, k_max=a.k_max, schedule="cosine",
                            annealed=bool(a.trainable_thr)) if a.n_bits is not None else None),
        "thresholds_learned_e": thresholds_learned,
        "threshold_trace": anneal_cb.history if anneal_cb is not None else None,
        "noise": dict(mu=a.noise_mu, train_sigma_e=a.noise_sigma, test_sigma_e=eval_noise,
                      train_test_mismatch=bool(a.test_noise_sigma is not None),
                      order="noise added to analog charge in the generator, then DigitizeLayer",
                      sigma_noise_nominal_e=SIGMA_NOISE_E),
        "noise_diagnostics": noise_diag,
        "labels_scale": dict(mode=a.label_scale_mode, train=list(scale_tr), eval=list(scale_ev),
                             dumped=dump_scale, train_over_eval=[float(v) for v in ratio]),
        "metrics": metrics,
        "router_hidden": list(a.router_hidden), "router_temp": a.router_temp,
        "total_params": int(model.count_params()),
        "per_expert_params": int(core.experts[0].count_params()),
        "router_params": int(n_router),
        "epochs_p1": len(h1.history["loss"]), "epochs_p2": len(h2.history["loss"]),
        "epochs_p3": len(h3.history["loss"]),
        "best_val_loss_p3": float(min(h3.history.get("val_loss", [float("inf")]))),
        "router_usage_val": (usage.tolist() if usage is not None else None),
        "router_entropy_val": ent,
        "paper_reference": PAPER_REF,
    }
    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1, default=float)
    print(f"\nwrote {pq_path}")
    print(f"wrote {os.path.join(out_dir, 'summary.json')}")
    print(f"ARM COMPLETE: {tag}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

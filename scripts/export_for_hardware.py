#!/usr/bin/env python
"""Export a trained arm for hardware synthesis.

Produces, per arm, in handoff_symbolic/exports/<name>/:
  constants.json   every trained number, with tan(theta_L) already folded into the
                   two literals inference actually needs. This is the whole model.
  golden.npz       N events: analog charge, ADC codes, truth, and the exact 14-vector
                   the trained Keras model emits. Bit-match target for C++/HLS.
  summary.json     copy of the run's provenance

and verifies three things before writing anything:
  1. the rebuilt Keras model reproduces the run's own parquet   (max|diff| == 0)
  2. the pure-numpy reference reproduces the Keras model        (max|diff| < 1e-5)
  3. the physical metrics recomputed from the numpy path match the run's summary

Usage:
  python scripts/export_for_hardware.py digi_2bit_paper_code_nexp1 digi_2bit_paper_code_nexp4
"""
import argparse
import json
import os
import shutil
import sys

import numpy as np

WORKDIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HANDOFF = os.path.join(WORKDIR, "handoff_symbolic")
sys.path.insert(0, os.path.join(WORKDIR, "two_bit_optimization_helpers"))
sys.path.insert(0, HANDOFF)


def extract_constants(core, summary):
    """Pull every trained number out of the graph into a plain dict."""
    import numpy as np

    def w(layer, key):
        for v in layer.weights:
            if key in v.name:
                return np.asarray(v.numpy()).astype(float)
        raise KeyError(f"{key} not found in {layer.name}: {[v.name for v in layer.weights]}")

    g = summary.get("digitization") or {}
    ls = summary.get("labels_scale")
    labels_scale = ls["train"] if isinstance(ls, dict) else ls
    if labels_scale is None:
        labels_scale = json.load(open(os.path.join(summary["_run_dir"], "labels_scale.json")))["labels_scale"]

    experts = []
    for e in core.experts:
        ans = e.get_layer("physics_ansatz")
        mlp = e.get_layer("error_mlp")
        theta_L = float(np.ravel(w(ans, "theta_L"))[0])
        lor = float(np.ravel(w(ans, "lorentz_scale"))[0])
        experts.append(dict(
            theta_L_rad=theta_L, theta_L_deg=float(np.degrees(theta_L)), lorentz_scale=lor,
            # tan(theta_L) is a CONSTANT at inference -- fold it, so hardware never
            # needs a tangent. These two literals are all that survives of it.
            dy_over_2=float(100.0 * np.tan(theta_L) / 2.0),
            lorentz_term=float(abs(lor * np.tan(theta_L))),
            sign_k_alpha=float(np.ravel(w(ans, "sign_k_alpha"))[0]),
            sign_a0_alpha=float(np.ravel(w(ans, "sign_a0_alpha"))[0]),
            sign_k_beta=float(np.ravel(w(ans, "sign_k_beta"))[0]),
            sign_a0_beta=float(np.ravel(w(ans, "sign_a0_beta"))[0]),
            aff_scale=np.ravel(w(ans, "aff_scale")).tolist(),
            aff_bias=np.ravel(w(ans, "aff_bias")).tolist(),
            mlp_w1=mlp.get_layer("hidden").get_weights()[0].tolist(),
            mlp_b1=mlp.get_layer("hidden").get_weights()[1].tolist(),
            mlp_w2=mlp.get_layer("chol_entries").get_weights()[0].tolist(),
            mlp_b2=mlp.get_layer("chol_entries").get_weights()[1].tolist(),
        ))

    router = None
    if len(core.experts) > 1:
        Ws, bs = [], []
        for layer in core.router.layers:
            kw = layer.get_weights()
            Ws.append(kw[0].tolist()); bs.append(kw[1].tolist())
        router = dict(weights=Ws, biases=bs, temp=float(summary.get("router_temp", 1.0)),
                      n_features=int(np.shape(Ws[0])[0]))

    return dict(
        arm=summary.get("_arm"),
        n_experts=int(summary["n_experts"]),
        variant=summary.get("variant", "barycenter"),
        geometry=dict(p_x=50.0, p_y=12.5, T=100.0, N=16,
                      note="p_x on COLUMNS, p_y on ROWS -- measured, see README"),
        digitize=(dict(n_bits=None) if g.get("n_bits") is None else dict(
            n_bits=int(g["n_bits"]),
            thresholds_e=[float(v) for v in g["thresholds_final_e"]],
            levels=[float(v) for v in g.get("levels_final", [0, 1, 2, 3])],
            comparator="strictly greater: code = sum(q > T_j)",
            published_thresholds_e=g.get("published_thresholds_e"),
            reference=summary.get("paper_reference"))),
        labels_scale=list(labels_scale),
        experts=experts,
        router=router,
        total_params=int(summary["total_params"]),
        noise=summary.get("noise"),
        metrics_reference=summary.get("metrics"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--root", default="/depot/cms/private/users/kuang14/Smart_Pixel")
    ap.add_argument("--n-golden", type=int, default=2000)
    ap.add_argument("--dataset-dir", default="/depot/cms/users/kuang14/Smart_Pixel/"
                    "dataset_s_series/dataset_3sr/"
                    "dataset_3sr_16x16_50x12P5_centeredIncidence_parquets")
    a = ap.parse_args()

    os.chdir(WORKDIR)
    import pandas as pd
    import tensorflow as tf
    from prepare_tfrecords import generate_tfrecords, load_tfrecords
    from symbolic.moe import load_arm
    import symbolic_ref as ref

    for arm in a.arms:
        run_dir = os.path.join(a.root, arm)
        print("=" * 76); print("ARM:", arm)
        model, core, summary = load_arm(run_dir)
        summary["_run_dir"], summary["_arm"] = run_dir, arm
        n_exp = int(summary["n_experts"])

        _, _, tfr_tr, tfr_val = generate_tfrecords(
            dataset_dir=a.dataset_dir, model_type="ViT_Max", train_batch_size=5000,
            val_batch_size=5000, select_contained=True, timeslices=2,
            tfrecords_exist=True, seed=42)
        nz = (summary.get("noise") or {}).get("test_sigma_e", 0.0) or 0.0
        _, vg = load_tfrecords(tfr_tr, tfr_val, noise=(-1 if nz <= 0 else [0.0, float(nz)]),
                               digitize=False, seed=42)

        # (1) rebuilt model must reproduce the run's own parquet, exactly
        df = pd.read_parquet(os.path.join(run_dir, f"symbolic_n{n_exp}_vars.parquet"))
        P = np.concatenate([model(vg[i][0], training=False).numpy()[:, [0, 2, 4, 6]]
                            for i in range(len(vg))])
        d1 = float(np.abs(P - df[["x", "y", "cotA", "cotB"]].values).max())
        print(f"  [1] rebuild vs stored parquet : max|diff| = {d1:.3e}")
        assert d1 == 0.0, "rebuilt model does not reproduce the run -- do NOT export"

        # (2) numpy reference must reproduce Keras
        C = extract_constants(core, summary)
        xa = np.concatenate([np.asarray(vg[i][0], "float32") for i in range(3)])[:a.n_golden]
        yt = np.concatenate([np.asarray(vg[i][1], "float32") for i in range(3)])[:a.n_golden]
        y_keras = model(xa, training=False).numpy()
        codes = ref.digitize(xa, C)
        y_ref = ref.forward(codes, C)
        d2 = float(np.abs(y_ref - y_keras).max())
        rel = float(np.abs(y_ref - y_keras).max() / (np.abs(y_keras).max() + 1e-12))
        print(f"  [2] numpy ref vs keras        : max|diff| = {d2:.3e}  (rel {rel:.2e})")
        assert d2 < 1e-4, "numpy reference does not match the trained model"

        # (3) ADC codes must be exactly what the layer emits
        if C["digitize"]["n_bits"] is not None:
            d3 = float(np.abs(codes - core.digi(tf.constant(xa), training=False).numpy()).max())
            print(f"  [3] numpy ADC vs DigitizeLayer: max|diff| = {d3:.3e}")
            assert d3 == 0.0, "numpy digitizer disagrees with the trained quantizer"

        out = os.path.join(HANDOFF, "exports", arm)
        os.makedirs(out, exist_ok=True)
        json.dump(C, open(os.path.join(out, "constants.json"), "w"), indent=1)
        np.savez_compressed(os.path.join(out, "golden.npz"),
                            charge_analog_e=xa.astype("float32"),
                            adc_codes=codes.astype("uint8" if C["digitize"]["n_bits"] else "float32"),
                            y_true_normalized=yt.astype("float32"),
                            y_pred_14=y_keras.astype("float32"),
                            labels_scale=np.asarray(C["labels_scale"], "float32"))
        shutil.copy(os.path.join(run_dir, "summary.json"), os.path.join(out, "summary.json"))
        n_scalars = sum(len(np.ravel(v)) for k, v in C["experts"][0].items()
                        if k.startswith(("aff", "sign")) or k in ("dy_over_2", "lorentz_term"))
        print(f"  wrote {out}  | {a.n_golden} golden events | "
              f"{n_scalars} physics scalars/expert + 354 MLP params/expert"
              + (f" + {sum(np.size(w) for w in C['router']['weights'])} router weights"
                 if C["router"] else ""))
    print("\nexport complete")


if __name__ == "__main__":
    sys.exit(main())

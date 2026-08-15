#!/usr/bin/env python
"""Check the numpy reference against the golden vectors. numpy only -- no TF.

    python verify.py                 # both arms
    python verify.py exports/digi_2bit_paper_code_nexp1

Use this as the template for your hardware check: swap `forward()` for your C++/HLS
output on the same `adc_codes` and compare against the same `y_pred_14`.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from symbolic_ref import digitize, forward, decode      # noqa: E402


def i68(r):
    """the group's resolution metric: half-width of the minimal 68% interval"""
    r = np.sort(np.asarray(r)[np.isfinite(r)])
    n = len(r); k = int(np.ceil(0.68 * n))
    return (r[-1] - r[0]) / 2.0 if k >= n else (r[k:] - r[:n - k]).min() / 2.0


def check(d):
    C = json.load(open(os.path.join(d, "constants.json")))
    g = np.load(os.path.join(d, "golden.npz"))
    ls = g["labels_scale"]
    name = os.path.basename(d)
    print(f"\n=== {name}  ({C['n_experts']} expert(s), {C['total_params']} params)")

    # 1. the ADC: analog charge -> codes
    codes = digitize(g["charge_analog_e"], C)
    d_adc = float(np.abs(codes - g["adc_codes"].astype("float32")).max())
    print(f"  ADC codes        : max|diff| = {d_adc:.3e}")

    # 2. the network: codes -> 14-vector
    y = forward(g["adc_codes"].astype("float32"), C)
    d_y = float(np.abs(y - g["y_pred_14"]).max())
    print(f"  14-vector        : max|diff| = {d_y:.3e}   (float32 eps ~ 1e-6)")

    # 3. the physics: same events, same metric as the software pipeline
    mu, sig = decode(y, ls)
    yt = g["y_true_normalized"] * ls
    print(f"  {'output':<6}{'I68':>10}{'unit':>6}{'sign acc':>11}")
    for i, (nm, unit) in enumerate([("x", "um"), ("y", "um"), ("cotA", "-"), ("cotB", "-")]):
        if unit == "um":
            t, p = yt[:, i], mu[:, i]
        else:                                   # angles compared in degrees
            t = np.degrees(np.arctan2(1.0, yt[:, i]))
            p = np.degrees(np.arctan2(1.0, mu[:, i]))
            unit = "deg"
        sa = ("" if i < 2 else
              f"{(np.sign(mu[:, i]) == np.sign(yt[:, i])).mean():11.4f}")
        print(f"  {nm:<6}{i68(t - p):10.3f}{unit:>6}{sa}")

    ok = d_adc == 0.0 and d_y < 1e-4
    print(f"  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    dirs = sys.argv[1:] or sorted(glob.glob(os.path.join(here, "exports", "*")))
    assert dirs, "no exports found -- run scripts/export_for_hardware.py first"
    sys.exit(0 if all([check(d) for d in dirs]) else 1)

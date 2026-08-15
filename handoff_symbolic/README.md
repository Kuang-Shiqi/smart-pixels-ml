# Symbolic student — hardware handoff

Two trained models, the closed-form expressions behind them, a dependency-free numpy
reference, and golden vectors to bit-match against.

Branch `symbolic` of `github.com/Kuang-Shiqi/smart-pixels-ml`; this directory is
self-contained apart from the two `.weights.h5` files, which live with their runs.

## Start here

```
handoff_symbolic/
  FORMULA.md          the model written out, with the trained numbers substituted
  symbolic_ref.py     pure-numpy forward pass -- no TF, no Keras, no qkeras
  verify.py           checks symbolic_ref against the golden vectors
  example.ipynb       load, verify, plot residuals, print the formula
  exports/
    digi_2bit_paper_code_nexp1/    constants.json, golden.npz, summary.json
    digi_2bit_paper_code_nexp4/    ditto
```

```
python verify.py                 # numpy only; should print PASS for both arms
```

## The two models

| | N=1 | N=4 (MoE) |
|---|---|---|
| params | **376** | 3272 |
| I68 x | 22.3 um | **6.57 um** |
| I68 y | 5.35 um | **2.19 um** |
| I68 alpha | 1.88 deg | **1.81 deg** |
| I68 beta | 2.44 deg | **2.29 deg** |
| sign accuracy alpha / beta | 0.973 / 0.971 | 0.974 / 0.972 |
| transcendental ops | 2 tanh | 2 tanh x4, plus sqrt, log, softmax in the router |

Both are trained at the deployment point: **2-bit input, published thresholds
248 / 668 / 1663 e- (arXiv:2602.15946 Fig. 3), code levels 0..3, no input noise.**

**Suggested order: N=1 first.** It is the whole pipeline — ADC, profiles, barycenter,
width, sign gate, output affine — with no square root, logarithm, exponential or
trigonometry in the mean path. Getting it through the flow validates the datapath and
the fixed-point choices. N=4 then only adds the router, which is where `sqrt`, `log`
and `softmax` appear. N=4 is the better model (3.4x on x) and the one we would want on
the chip, so it is the target; N=1 is the on-ramp.

## Interfaces

**Input** `(16, 16, 2)` = [row, col, time slice]. **row -> y at 12.5 um, col -> x at 50 um.**
Measured on 50k events, not assumed — an earlier version of this model had the two axes
transposed, which cost a factor 12 on the alpha resolution and was invisible in the
position outputs because a wrapper was silently correcting them. If anything looks
transposed in your port, that is the first thing to check: light one pixel at
(row 0, col 15) and the x output must read +375 um.

**Output** `(14,)` = `[x, s_x, y, s_y, cotA, s_A, cotB, s_B, M21, M31, M32, M41, M42, M43]`
— means at even indices 0/2/4/6 in **normalized** units (multiply by `labels_scale`),
raw Cholesky diagonals at 1/3/5/7 (softplus them to get sigma), 6 off-diagonals last.
`decode()` in `symbolic_ref.py` does it.

**ADC boundary — the one thing to decide before you start.** `constants.json` carries the
three comparator thresholds in electrons, and `golden.npz` gives you both sides:
`charge_analog_e` (before) and `adc_codes` (after). So you can put your hardware boundary
either place. Our assumption is that the ADC is on-sensor and the network receives codes
0..3, i.e. the three thresholds are comparators, not part of the datapath. Tell us if your
flow wants it the other way and we will export accordingly.

## Golden vectors

`golden.npz`, 2000 events each:

| key | shape | |
|---|---|---|
| `charge_analog_e` | (2000, 16, 16, 2) | input charge in electrons, before the ADC |
| `adc_codes` | (2000, 16, 16, 2) uint8 | after the ADC — 0..3, the network's actual input |
| `y_pred_14` | (2000, 14) | what the trained Keras model emits. **Bit-match target** |
| `y_true_normalized` | (2000, 4) | truth, normalized. Multiply by `labels_scale` for um |
| `labels_scale` | (4,) | 123.410162, 30.929850, 6.577499, 1.929565 |

The numpy reference reproduces `y_pred_14` to **9.5e-7 (N=1)** and **2.6e-6 (N=4)** max
absolute difference — that is float32 rounding, so treat `y_pred_14` as exact.

## What "matching" means

The physics tolerance is much looser than bit-exactness. Quantization of the *inputs* to
2 bits already costs ~8% on alpha and ~43% on x relative to analog, so a hardware port
that lands within **1%** of the software I68 is doing fine. Suggested acceptance:

1. `max|y14_hw - y14_sw|` on the golden set, reported per output slot — diagnostic, not a gate.
2. I68 of the residual per output, hardware vs software, on the same 2000 events. Within 1%.
3. **Sign accuracy for alpha and beta.** This is the fragile one: the gate reads a
   *difference of two nearly equal centroids*, so it is where fixed-point precision will
   bite first, long before the positions move. Software values are 0.973 / 0.971.

## Regenerating any of this

```
python scripts/export_for_hardware.py digi_2bit_paper_code_nexp1 digi_2bit_paper_code_nexp4
```

It refuses to write unless (1) the rebuilt Keras model reproduces the run's own parquet
exactly, (2) the numpy reference matches Keras, and (3) the numpy ADC matches the trained
quantizer bit-for-bit.

To rebuild the Keras model yourself (needs TF 2.15 + qkeras — the `SmartPixels` kernel):

```python
import sys; sys.path.insert(0, "two_bit_optimization_helpers")
from symbolic.moe import load_arm
model, core, summary = load_arm("/depot/cms/private/users/kuang14/Smart_Pixel/digi_2bit_paper_code_nexp4")
y14 = model(x).numpy()
```

## Things worth knowing before you port

- **`theta_L` is a constant at inference.** It is folded into two literals (`dy_over_2`,
  `lorentz_term`) in `constants.json`. No tangent in hardware.
- **`tanh` only needs to be accurate near zero.** It is a sign gate: the argument is
  `k*(a0 - drift)` with `k ~ 0.2 /um`, so it saturates for most events and the interesting
  region is a few um wide around the boundary. A coarse LUT should be fine, but it is worth
  sweeping — this is the observable that carries the sign.
- **The `+1e-6` in every denominator matters.** Empty or near-empty profiles are common;
  those guards are what keep the barycenters finite.
- **The width counts are the only comparisons** in the mean path, and they are what set both
  angle magnitudes. Their behaviour at the cluster edge is what 2-bit quantization degrades.
- The uncertainty MLP does not feed the means and can be dropped if the covariance is not
  needed on-chip. That leaves 22 physics parameters for N=1.

## Contact points in the repo

| what | where |
|---|---|
| ansatz (the physics) | `two_bit_optimization_helpers/symbolic/ansatz.py` |
| model assembly, weight loading | `two_bit_optimization_helpers/symbolic/moe.py` |
| ADC / SoftQuantize front-end | `two_bit_optimization_helpers/symbolic/digitize.py` |
| training (one arm per invocation) | `scripts/train_arm.py`, see `scripts/RUNBOOK.md` |
| comparison plots | `compare_baselines_vs_symbolic.ipynb` |

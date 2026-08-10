# Symbolic Regression Update — roundtable outline

Successor to `previous_slides/Symbolic Regression updates.pdf` (7/27).
That deck closed with three "next steps": **expert-count comparison**, **fix the KD bug**, and
**performance under input digitization & noise**. This deck delivers #1 and most of #3.

**Suggested length:** 12 content slides + 4 backup. Numbers below are copy-paste ready.

---

## One-line framing for the whole talk

> The closed-form student now runs on a **2-bit ADC input, which is the real on-sensor
> constraint, not an ablation arm** — and quantization costs less than expected, while the sign
> head that we built in July survives it intact.

---

## Slide 1 — Recap (30 seconds, do not re-teach July)

- Symbolic distillation: ~400k-param ViT → closed-form physics formula, ~370 params.
- July result: cluster width is **even** in incidence angle, so the ansatz only recovers
  `|cotα|`, `|cotβ|`. The sign lives in the **space-EVEN × time-ODD** sector, and a 2-parameter
  gate on the between-slice centroid drift extracts it:
  `sign(cotα) = tanh(k·(a₀ − Tx))`.
- **Visual:** reuse the symmetry-sector / toy-example slide from the 7/27 deck verbatim.

## Slide 2 — What's new since 7/27

Three bullets, then spend the talk on them:

1. Found and fixed an **axis-convention bug** inside the ansatz (was masked by a wrapper).
2. Answered the **how-many-experts** question.
3. Built **input digitization into the model** and ran the bit-depth study.

---

## Slide 3 — Bug: the ansatz had rows and columns swapped

The ansatz built its x output from the **row** axis (which tracks true y) and vice versa. It had
been patched from outside by an `AxisFixExpert` wrapper; the fix is now inside `ansatz.py` (v3) and
also corrects the **pitches, cluster widths, and the sign observables**, not just the two position
slots.

Diagnostic that catches it in one line: a fresh untrained expert gives `corr(x_pred, x_true) ≈ −0.003`.

| N=1 analog | x µm | y µm | cotA ° | cotB ° |
|---|---|---|---|---|
| wrapper patch only (v2) | 15.56 | 4.11 | **20.31** | 4.58 |
| fixed in-ansatz (v3) | 15.61 | 3.75 | **1.74** | 2.56 |

- cotA improves **11.7×**. The wrapper never reached the width/pitch terms that cotA is built from.
- **Be honest on this slide:** for the 4-expert model the same comparison gives cotA 1.39 → 1.64,
  i.e. slightly *worse*. The MoE router had learned to compensate for the swap. See Slide 5 caveat —
  that N=4 point also needs a re-run for an unrelated reason, so do not over-interpret it yet.
- All numbers are **I68** (half-width of the minimal 68% interval), the paper metric.
- **Visual:** `axis_convention_diagnostic.ipynb` cell 3 (text output — screenshot the correlation
  table), or just state the −0.003 number.

## Slide 4 — How many experts? (answers July next-step #1)

| model | params | x µm | y µm | cotA ° | cotB ° |
|---|---|---|---|---|---|
| Single formula, N=1 | **372** | 15.61 | 3.75 | 1.74 | 2.56 |
| MoE, N=4 | 3268 | **6.51** | **2.50** | **1.64** | **2.21** |
| Conv2D Full (paper) | 1767 | 4.14 | 1.00 | 1.04 | 1.20 |
| MLP Full (paper) | 2264 | 4.05 | 0.92 | 0.89 | 0.88 |

- Going N=1 → N=4 buys **2.4× on x** for ~9× the parameters. Angles barely move.
- Router usage is healthy, not collapsed: `[0.24, 0.05, 0.40, 0.32]`, entropy 1.20 / ln4 = 0.86.
- Honest position: N=4 is **within ~1.6× of the paper baselines on x** at comparable parameter
  count; N=1 is not, but N=1 is a *single closed-form expression*, which is the point.
- **Visual:** `compare_baselines_vs_symbolic.ipynb` **cell 9** — params-vs-resolution scatter,
  bottom-left = better. This is the single best slide in the deck; make it full-bleed.

---

## Slide 5 — Digitization is inside the model, not a preprocessing step

Say why this is not just "quantize the input array":

- The ansatz is built from **charge-weighted moments** — centroids, widths, third moments. Coarse
  quantization attacks exactly the dynamic range those moments need.
- The sign gate reads **`Tx`, `Ty` = between-slice centroid drift**, a small *differential*
  quantity. Before running it, this was the piece we expected to break.
- Quantization is non-differentiable → **straight-through estimator**, hard quantize on the forward
  pass at all times, sigmoid-CDF surrogate on the backward pass.

Implementation: `DigitizeLayer` (`two_bit_optimization_helpers/symbolic/digitize.py`), sitting
**ahead of both the router and the experts**, so router and experts see the identical digitized
tensor. Thresholds are in **physical charge units (electrons)**, monotone by construction.
`n_bits=None` is an identity passthrough, so the analog point runs the *same graph* — it is not a
different model.

Two design knobs worth naming out loud, because they are ASIC decisions:

- **Threshold placement** — `lowfirst` pins T0 just above the noise floor (100 e⁻) and spreads the
  rest by occupancy. Plain equal-occupancy erodes the cluster edge, which is what sets `|cotβ|` on
  the fine 12.5 µm axis.
- **Level mode** — `code` = the ADC emits 0…3 (cheapest chip). `midpoint` = a 4-entry code→charge
  LUT on-chip.

## Slide 6 — Level mode is not free: the LUT pays for itself in sign accuracy

Measured with **no training at all** (`sign_vs_bitdepth.ipynb`), so this is a property of the data,
not of a fit:

| 2-bit, `lowfirst` | sign acc α | sign acc β | logistic ceiling α |
|---|---|---|---|
| `code` (0…3, no LUT) | 0.9467 | 0.9105 | 0.9399 |
| `midpoint` (4-entry LUT) | **0.9724** | **0.9619** | 0.9433 |

- The LUT is worth **+2.6 pts on cotα sign, +5.1 pts on cotβ sign**. Four numbers of on-chip
  storage.
- Mechanism: `code` levels compress the charge ratio that the centroid drift is computed from —
  mean |Tx| drops from 75 µm (`midpoint`) to 32 µm (`code`).
- **Visual:** this table. Optionally add mean-|Tx| as a second row; the collapse is the mechanism.

## Slide 7 — HEADLINE: the cost is analog → digitized, and then it is flat

Trained end-to-end through the quantizer, N=1, `lowfirst`/`midpoint`:

| | x µm | y µm | cotA ° | cotB ° |
|---|---|---|---|---|
| analog | 15.61 | 3.75 | 1.744 | 2.563 |
| 4-bit | 20.14 | 5.05 | 1.811 | 2.749 |
| 3-bit | 20.02 | 5.11 | 1.817 | 2.918 |
| **2-bit** | **20.08** | **5.05** | **1.829** | **2.976** |

- Analog → digitized costs **+29% x, +35% y, +5% cotA, +16% cotB**.
- 4-bit → 2-bit costs **essentially nothing**: x and y agree to <1%, cotA to 1%.
- **The message for the ASIC side: 2 bits already buys everything 4 bits does.** There is no
  bit-depth trade to argue about, which is convenient, because 2 bits is what we get.
- **Say this if it comes up:** an earlier version of this table used residual *std*, where x looked
  non-monotone (20.4 at 2-bit vs 22.1 at 4-bit). That was tail-driven. In I68 the inversion
  disappears. Quote I68.
- **Visual:** `compare_baselines_vs_symbolic.ipynb` **cell 5** (residual panels, 2×2:
  x / y / cotA / cotB). Filter `FILES` down to the four N=1 points first so the panel is readable —
  11 overlaid curves is unreadable on a projector.

## Slide 8 — The sign head survives quantization (the thing we expected to break)

| | sign acc α | sign acc β |
|---|---|---|
| analog | 0.9672 | 0.9674 |
| 4-bit | 0.9755 | 0.9687 |
| 3-bit | 0.9751 | 0.9681 |
| **2-bit** | **0.9728** | **0.9687** |

- Raw-pixel logistic regression on **analog** data gets 0.9662 / 0.9663 — that is the ceiling. The
  2-bit model **matches the analog ceiling**.
- Sign accuracy at 2-bit is *slightly higher* than analog. Do not oversell this; the honest reading
  is that thresholding suppresses low-charge pixels that add noise to the drift estimate, and the
  effect is ~0.5 pt.
- This is the slide that closes the July story: the drift-based sign gate is not a fragile trick,
  it is robust to the real front-end.

## Slide 9 — Calibration: are the uncertainties honest?

The student predicts a full 4×4 covariance (10 Cholesky entries), so pulls must be checked, not
assumed.

- Gaussian-fit pull σ, 2-bit N=1: x 0.995, y 0.828, cotA 0.770, cotB 0.628.
- Pull means are ~0 everywhere (|µ| < 0.06) — **no bias**.
- σ < 1 means the model is **conservative** (over-estimates its own error), worst on cotB.
  Under-coverage would be the bad failure; this is the safe direction, but it is not perfect and we
  should say so.
- **Visual:** `compare_baselines_vs_symbolic.ipynb` **cell 6** — log-scale step histograms with
  Gaussian overlay, plus the printed fit table.

## Slide 10 — Interpretability: what the formula actually learned

Everything the model knows is in ~10 named scalars. Two of them tell a story:

| scalar | analog | 2-bit |
|---|---|---|
| `lorentz_scale` | 0.00013 (**off**) | 0.051 (**on**) |
| `theta_L` | 0.065 rad | 0.360 rad ≈ 20.6° |

- With analog charge the fit **switches the Lorentz drift correction off** — the raw moments are
  informative enough. Under 2-bit it **switches it on**: when information is scarce, the physics
  prior becomes load-bearing.
- Caveat to state: with `lorentz_scale ≈ 0` the analog `theta_L` is unconstrained, so this is an
  observation about which terms the fit uses, **not** a measurement of the Lorentz angle.
- Sign-gate offset `a₀β ≈ −3.9 to −7.5 µm` — a plausible E×B drift signature, same sign as in July.
- **Visual:** none needed; a two-column table of the scalars is stronger than a plot. Source:
  `expert_scalars.json` in each run directory, printed by `train_digitization_ablation.ipynb` cell 11.

---

## Slide 11 — What is running now

Be explicit that this is in flight, not done:

- **MoE N=4 at 2-bit** — queued. Every digitized run so far is N=1. This matters: quantization
  costs ~4.5 µm on x, but N=1 → N=4 is worth ~9 µm. The deployment candidate is almost certainly
  N=4 at 2 bits, and it has never been trained.
- **Known issue to disclose:** the existing `symbolic_axisfix_nexp4` analog point was trained with
  the *crossed* sign wiring (`α←Ty, β←Tx`) before that convention was corrected — it predates the
  N=1 run by 12 hours. So the N=4 analog reference is stale and needs a re-run from the current
  notebook before the MoE analog↔2-bit comparison is apples-to-apples. Both runs go through
  `train_digitization_ablation.ipynb` so they share one code path.
- **`level_mode='code'` trained end-to-end** — Slide 6 shows the untrained cost; we have not yet
  checked how much training recovers.
- **Trainable thresholds.** At 2 bits there are exactly **three** thresholds and they *are* the ADC
  design. `DigitizeLayer` supports co-training them; it has never been switched on. The deliverable
  is three charge values handed to the ASIC team — this is the most concretely useful thing on the
  list.

## Slide 12 — Next / asks

- **Noise (July next-step #3, second half).** Order is physically fixed: noise is added to the
  analog charge, *then* digitized. The generator already does this in the right order. Grid:
  {analog, 2-bit} × {none, low, nominal, high}, plus train-clean/test-noisy to quantify what
  noise-aware training actually buys.
- **KD (July next-step #2)** — still blocked. The ViT teacher was trained on 2-bit input but its
  `labels_scale` (cotA/cotB ≈ 9.3 / 4.3) does not match the current tfrecords (6.577 / 1.930). KL is
  disabled until a runtime rescale is applied. *Ask the room:* does anyone have the exact scaling
  used for the teacher's training set?
- **Hardware synthesis** — the axis fix is now inside `ansatz.py`, which was the blocker for
  hls4ml/HLS export. Worth asking whether anyone wants to try the N=1 (372-param) model through
  hls4ml now.

---

## Backup slides

- **B1** — full 12-model resolution table (`compare_baselines_vs_symbolic.ipynb` cell 7 printout).
- **B2** — the full `sign_vs_bitdepth.json` grid: 13 rows, {analog, 4, 3, 2} × {quantile, lowfirst,
  linear} × {code, midpoint}, with logistic ceilings. Good defence against "did you tune the
  thresholds to get that answer".
- **B3** — actual 2-bit thresholds and levels used, in electrons:
  thresholds `[100.0, 240.1, 1295.0]`, levels `[0, 170.1, 767.6, 5856.5]`, offset 80 e⁻.
- **B4** — training schedule, in case anyone asks why three phases: phase 1 MSE on means (150 ep) →
  phase 2 NLL refine (≈200 ep) → phase 3 error-head only (120 ep). Starting from NLL lets σ-inflation
  kill the mean gradients.

---

## Plot-generation checklist

Everything visual comes from `compare_baselines_vs_symbolic.ipynb`. The training notebooks emit
text only.

| Slide | Notebook | Cell | What |
|---|---|---|---|
| 4 | `compare_baselines_vs_symbolic.ipynb` | 9 | params vs resolution scatter |
| 7 | `compare_baselines_vs_symbolic.ipynb` | 5 | residual panels, 2×2 |
| 9 | `compare_baselines_vs_symbolic.ipynb` | 6 | pull histograms + fit table |
| B1 | `compare_baselines_vs_symbolic.ipynb` | 7 | resolution table |
| 3 | `axis_convention_diagnostic.ipynb` | 3 | correlation diagnostic (text) |
| 6, B2 | `sign_vs_bitdepth.ipynb` | 3 | sign-vs-bit-depth grid (text → `sign_vs_bitdepth.json`) |

**Before generating slide figures:** cut `FILES` down per figure. Slide 4 wants baselines + N=1 +
N=4 analog. Slide 7 wants only the four N=1 bit-depth points. The current 12-entry list is right for
the resolution table and wrong for every plot. Set `output_dir` in cell 2 to save PNGs instead of
screenshotting.

## Numbers you should be able to defend without slides

- 372 params (N=1), 3268 (N=4). Paper baselines 1.8k–2.7k.
- 108,222 validation events, all tables.
- Analog → 2-bit: +29% x, +35% y, +5% cotA, +16% cotB in I68.
- Sign accuracy 0.973 / 0.969 at 2-bit; analog raw-pixel logistic ceiling 0.966.
- 2-bit thresholds: 100 / 240 / 1295 e⁻ above an 80 e⁻ offset.

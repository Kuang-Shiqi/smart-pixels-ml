    # CLAUDE.md — SmartPixels symbolic student: input digitization + noise robustness

## 0. How to work in this repo

**TRAINING RUNS FROM SCRIPTS. PLOTS RUN FROM NOTEBOOKS. Do not mix the two.**
(Decided 2026-08-14. Full detail in `scripts/RUNBOOK.md`.)

- **Training = `scripts/train_arm.py`**, one arm per invocation, every knob a flag, every flag
  recorded in `summary.json`. You can run it yourself:
  `/work/users/kuang14/smart_pixels/bin/python scripts/train_arm.py --n-bits 2 ...`
  Use `--dry-run` first. `scripts/run_queue.sh` chains arms sequentially.
- **Why.** A notebook open in JupyterLab keeps its own in-memory copy and overwrites the file
  when it saves. This already cost a threshold-freeze fix (reverted silently in
  `train_noise_ablation.ipynb`) and a `NOISE_SIGMA` edit, which produced a run whose config did
  not match its own output directory. Scripts have one writer.
- **Never edit a notebook the user has open.** If a notebook edit is unavoidable, say so and tell
  them to `File > Reload Notebook from Disk` first.
- **Plots = notebooks**, because the user reads them interactively. Cell style there is unchanged:
  self-contained, imports at the top, runnable *in place*. No bolt-on patch cells at the bottom —
  fold fixes into the canonical cell.
- **One-off recovery in a live kernel:** write a `.py` next to the notebook and have the user run
  `%run -i thing.py`. `-i` shares the namespace, so it sees the trained model. Beats pasting a
  long cell out of the terminal, which they cannot easily copy.
- **Terse output.** Bulleted, copy-paste-ready. Skip the explanation unless asked. Do not restate
  the plan back before doing it.
- If the user says a bug is confirmed, it is confirmed. Do not re-litigate with alternative
  explanations.

### Environment (measured, not assumed)

| | |
|---|---|
| training interpreter | `/work/users/kuang14/smart_pixels/bin/python` — TF **2.15.1 / Keras 2.15** / qkeras. This is the `SmartPixels` Jupyter kernel |
| AF global pixi env | `/work/pixi/global/...` — pandas/numpy/matplotlib only, **no qkeras**, cannot train. Fine for reading parquets and running the plotting cells headlessly |
| GPU | one **A100 MIG 1g.5gb slice (5 GB)**, not a full A100. Run arms sequentially |
| Slurm | `sbatch`/`squeue` reachable; Hammer has 13 GPU nodes, 30-day walltime. **Blocked**: workers see only `/depot`, and the training env lives on `/work` |

## 1. Project in one paragraph

Silicon pixel detector hit reconstruction for CMS SmartPixels. Input: a 16x16 pixel charge array
with 2 time slices (shape `(16,16,2)`, analog by default). Output: 14 numbers = 4 track parameter
means `(x, y, cotA, cotB)` + 10 Cholesky entries of the 4x4 covariance. Target: on-sensor ASIC,
28 nm CMOS, sub-25 ns latency, so the student must be tiny (~370–3,300 params) and hls4ml/HLS
exportable. A 418k-param ViT_Max teacher exists for knowledge distillation (~130x compression).

Dataset: `dataset_3srb_16x16_50x12P5_centeredIncidence_10ps_300k_convolved_to_200ps`,
50 x 12.5 um pitch, tfrecords produced by `generate_tfrecords()`.

## 2. Where things live

| What | Path |
|---|---|
| Working root | `/depot/cms/private/users/kuang14/Smart_Pixel/` |
| Previous symbolic repo (reference, DO NOT edit) | `.../Smart_Pixel/smart-pixels-symbolic-updated/` |
| **Digitization reference folder (read-only reference)** | `/depot/cms/private/users/kuang14/Smart_Pixel/Arghya_input_digitization` |
| Arghya's teacher artifacts | `/work/users/das214/SmartPixels/smart-pixels-ml/` |
| Teacher checkpoint (the correct one) | `/work/users/das214/SmartPixels/smart-pixels-ml/runs/weights/weights-2t-ViT_Max-2bit_optimized-from_part1_extract-checkpoints/weights.01-t-43762.58-v-43762.58.hdf5` |
| GitHub | `github.com/Kuang-Shiqi/smart-pixels-ml`, branch `symbolic`, SSH `~/.ssh/id_ed25519` |

Core modules (in the symbolic repo):
- `two_bit_optimization_helpers/symbolic/ansatz.py` — the physics formula + sign head
- `two_bit_optimization_helpers/models/student_max.py` — `build_student_max(variant, ansatz_kwargs=...)`
- `two_bit_optimization_helpers/{distill,loss,train,prepare_tfrecords}.py`
- **Scripts (the canonical training path):**
  - `scripts/train_arm.py` — one arm, all knobs as flags. `--dry-run` to check first
  - `scripts/run_queue.sh` — chains arms sequentially, skips ones with a `logs/*.done` marker
  - `scripts/RUNBOOK.md` — commands, the arm table, and the gotchas that already bit us
- Notebooks (plots; the two training notebooks are now secondary, kept for interactive debugging):
  - `train_symbolic_nexp_ablation.ipynb` — N-expert ablation, analog input
  - `train_digitization_ablation.ipynb` — same student + `DigitizeLayer`. Clean input (sigma = 0)
  - `train_noise_ablation.ipynb` — duplicate of the above with `NOISE_SIGMA` on. Same
    `digi_*_nexp*` output convention, so both sets compare directly
  - `compare_baselines_vs_symbolic.ipynb` — all comparisons. Section A = axis fix before/after in
    isolation; section B = digitization/SoftQuantize, auto-discovers every `digi_*` run
  - `axis_convention_diagnostic.ipynb` — the measurement that found the transpose (§4.1). Its prose
    describes the *pre*-fix state; it is a record, not current

## 3. Model architecture (current, working)

```
inputs (16,16,2) analog
   |
   +-- RouterFeatures --> softmax router (N experts, ROUTER_TEMP)
   |
   +-- N x [ PhysicsAnsatz (v3 axes) ]          # unsigned |x|,|y|,|cotA|,|cotB|
   |        + TimeGradSignHead                  # supplies the SIGN
   |
   +-- error_mlp --> 10 Cholesky entries
   |
   v
 pack_14 --> 14-vector
```

- **N=1** collapses to a single closed-form physics formula (~370 params). **N=4** MoE ~3,260 params.
- Trainable scalars in the ansatz: Lorentz angle `theta_L`, `lorentz_scale`, per-output affine
  scale/bias.

### The sign head (the hard-won result — do not break this)

Cluster width is an **even** function of incidence angle, so the ansatz can only emit `|cotA|`,
`|cotB|`. Every time-summed spatial moment (skewness, dipoles, centroids) is provably blind to the
sign. The sign lives in the **spatially-even x time-odd** symmetry sector — i.e. the between-slice
centroid drift.

```
sign(cotA) = tanh( kA * (a0A - Tx) )     # UNCROSSED: each angle is signed by
sign(cotB) = tanh( kB * (a0B - Ty) )     # the drift along its OWN axis
```

where `Tx`, `Ty` = centroid drift between the two time slices; `k` and `a0` are trainable
(2 scalars per angle). `k` is negative-direction for both. Raw-pixel logistic regression gets
96.6% sign accuracy — that's the ceiling, and these gates sit at 0.967 / 0.963.

**The pairing is uncrossed, and this is not a change of physics.** v2 wrote it crossed
(`cotA <- Ty`) because it had the axes transposed — what v2 called `Ty` was already the *column*
drift, i.e. the x-drift. v3 fixed the names, so the same observable now reads as the uncrossed
pairing the physics expects. Do **not** "re-cross" it back. The other two pairings sit at chance
(0.504, 0.500), so a wrong choice here is loud, not subtle.

Calibration does not transfer across the axis fix: the drifts carry the pitch. Under v3,
`kA = 0.158 /um, a0A = +0.16 um`, `kB = 0.335 /um, a0B = -10.86 um`. **The old reading of `a0B`
as a Lorentz (ExB) drift signature did not survive the pitch correction** — against
`T*tan(22 deg) = 40 um`, -10.9 um is not a match. Flagged, unresolved; do not repeat the Lorentz
claim in a talk. Both stay trainable, and the training notebooks refit `(k, a0)` by logistic fit
on train data anyway, so the defaults are only a starting point.

**Historical trap:** the sign used to be borrowed from the teacher via `apply_sign` / `teacher_signs`.
That is dead. Do not reintroduce it. The teacher-borrowed sign was *anti*-correlated (0.35 agreement)
because of the input-encoding bug in §5.

## 4. Known bugs / invariants — read before writing any training code

1. **Axis swap — FIXED in `ansatz.py` (v3). Do not re-add a wrapper.** v2 had the two spatial axes
   transposed: it read x off the *rows*. Measured truth (`axis_convention_diagnostic.ipynb`, 50k val
   events): **cols -> x @ 50 um** (corr 0.927), **rows -> y @ 12.5 um** (corr 0.914), cross terms
   0.002. The axis choice carries three things, and the old `AxisFixExpert` wrapper patched only the
   first: (a) which output slot gets which profile, (b) the pitch attached to each profile — the
   x-profile was scaled by 50 um while indexing the 12.5 um rows, (c) the cluster widths that set
   `|cotA|`, `|cotB|`. v3 fixes all three inside the file, plus the sign observables.
   - **`AxisFixExpert` is deleted and must stay deleted** — with v3 it would double-swap and
     reproduce the original bug exactly.
   - Runtime guard, already in cell 5/6 of both training notebooks: assert
     `corr(x_slot, x_true) > 0.8`. A stale kernel, a v2 `ansatz.py`, or a re-added wrapper trips it
     before training starts.
   - What the fix bought (N=1): `alpha` I68 **20.31 -> 1.74 deg**, `beta` 4.58 -> 2.56,
     `y` 4.11 -> 3.75, `x` 15.56 -> 15.61 (flat), x residual mean -6.74 -> -0.03 um. It is an
     *angle* fix. **x is still ~15.6 um vs ~4 um for the paper baselines — untouched and open.**
   - `symbolic_ablation_*` parquets are v2+wrapper output and are the "before". Never regenerate
     them with v3.
2. **NaN from `std**3` underflow.** Narrow clusters -> `std` near zero -> `m3/std^3` blows up.
   Fixes: relu the charge inputs, **std floor = 1.0** (not 1e-9), softplus diagonal in the loss.
3. **Two-phase (three-phase) training for NLL.** Phase 1: MSE on means. Phase 2: NLL refine.
   Phase 3: error-head-only. Starting with NLL lets sigma-inflation kill the mean gradients.
4. **Loss/decode convention must match exactly.** `custom_loss` and the parquet dump cell must both
   use softplus diagonal and `mean(-log_prob)`. A previous mismatch (ReLU diag + `K.sum` with clip
   floor) zeroed gradients.
5. **`get_layer("physics_ansatz")` fails through any expert wrapper** (name-prefix mismatch). With
   v3 there is no wrapper, so the plain call works; keep `getattr(e, "inner", e)` where it already
   is, since it is a no-op now and still loads the v2-era runs.
6. **`TF_USE_LEGACY_KERAS=1` breaks training** (numerical incompatibility with Keras 3.x). Never set it.
6b. **`Variable.trainable` is READ-ONLY in the training env** (Keras 2.15 -> `add_weight` returns a
   `tf.Variable`). `threshold_deltas_raw.trainable = False` raises
   `AttributeError: can't set attribute`, while `.assign()` on the same variable works. Use
   `freeze_adc_thresholds()` (in `train_arm.py`): variable -> sublayer -> private flag, verified
   against `model.trainable_variables`, which is the only list the optimizer reads. In this env it
   takes the **sublayer** path.
6c. **This dataset has INTRINSIC negative charge** with the noise off: ~2.9% of pixels, minimum
   around -3400 e- (shaping undershoot in the convolved simulation). Any "negative charge implies
   injected noise" logic is wrong here. To measure an injected sigma, use
   `median(|q|, q<0)/0.6745` — robust to that tail, because real noise makes ~half of all pixels
   negative while the intrinsic tail is rare.
6d. **Train and eval splits carry DIFFERENT `labels_scale`** (99th percentile of |label|, computed
   per split): train `123.410162 / 30.929850 / 6.577499 / 1.929565`, eval
   `122.896897 / 30.903849 / 6.560915 / 1.917222`. The eval vector is byte-for-byte the
   `PAPER_SCALE` constant in the comparison notebook. The ansatz is fit against the TRAIN
   normalization and evaluated against the EVAL one, so there is a built-in slope of
   **+0.42% (x), +0.08% (y), +0.25% (cotA), +0.64% (cotB)** in every residual plot. Every run so
   far dumps the train vector (`--label-scale-mode train`, the default, for comparability).
   `--label-scale-mode align` rescales predictions into the eval normalization and removes it.
   Nothing in our conclusions turns on 0.4%, but do not quote sub-percent effects until this is
   settled, and do not mix modes across runs being compared.
7. **`os.walk` needs depth limiting** (`if depth > 4: dirs.clear()`) — checkpoint trees are huge.
8. **SLURM on Gilbreth** requires `--partition`, `--account`, `--qos`, and `--gpus-per-task`
   (mandatory even for CPU-only jobs).

## 5. Teacher / distillation status (deferred, but keep it un-broken)

- The ViT_Max teacher was **trained on 2-bit digitized inputs** but was being **queried on analog**
  throughout distillation. This produced anti-correlated sign predictions. Fixed by the
  `PairedGen` / `TwoBitTeacherDistiller` architecture: student gets analog `xa`, teacher gets 2-bit
  `xd`, from the same seed, with per-batch label alignment asserts.
  - `Distiller.call` passes `x` straight to the student, so `TwoBitTeacherDistiller` must override
    `.call` to unpack `(xa, xd)` and pass only `xa`; `_forward` uses both.
- **KL is currently disabled** (`WARMUP_STEPS = 10**9`) because of a `labels_scale` mismatch:
  teacher training scales cotA/cotB ~ 9.3 / 4.3 vs current tfrecords ~ 6.577 / 1.930. KL is
  meaningless until a runtime rescale is applied. `distillation_audit_tf.py` was written to compute
  the exact factors; it has not been run yet.
- Correct checkpoint = the `from_part1_extract` variant (24 layers, `soft_quantize_layer=False`,
  val loss -43762.58). The 30-epoch alternative is substantially worse. A 25-vs-24 layer load error
  means you grabbed the wrong one.

## 6. GOALS FOR THIS FOLDER (in priority order)

### Goal 1 — Input digitization inside the symbolic model

> **2-bit is a hard requirement, not one arm of an ablation.** The on-sensor ADC *is* 2-bit in the
> real detector. Analog / 4-bit / 3-bit are **reference points that quantify what quantization
> costs** — they are never candidate designs, and no result may be reported as "use N bits" for
> N != 2. Every model that is meant to ship trains and is evaluated at 2 bits. The open question is
> not *how many bits* but **how good can we get at 2 bits** (threshold placement, level mode,
> N_EXPERTS, noise-aware training).

The student currently trains on clean analog inputs. We need a **digitization layer at the front of
the model** and to work out how it interacts with the symbolic ansatz.

Why this is non-trivial here (not just a data-preprocessing step):
- The ansatz is built from **charge-weighted moments** (centroids, second moments, third moments).
  2-bit quantization crushes the dynamic range those moments depend on.
- The **sign head depends on between-slice centroid drift `Tx`, `Ty`** — a small differential
  quantity. This is the part most at risk from coarse quantization. **Measure sign accuracy
  explicitly as a function of bit depth.**
- Quantization is non-differentiable -> needs a straight-through estimator (STE) or a soft/annealed
  version during training. Check what `Arghya_input_digitization` already does before writing new code.

### PUBLISHED CONSTANTS — use these, never re-derive from our own data

Source: **arXiv:2602.15946 Sec. IV** ("Input charge digitization with end-to-end training"), the
group's own regression paper. Encoded in `two_bit_optimization_helpers/symbolic/digitize.py`; import
them, do not retype them.

| constant | value | where |
|---|---|---|
| `sigma_noise` | **80 e-** | Sec. IV B, Cadence Virtuoso design simulated with Spectre |
| 5 sigma (conventional lowest threshold) | **400 e-** | Sec. IV B |
| two-bit thresholds | **248 / 668 / 1663 e-** (+- 6 / 8 / 39) | Fig. 3, Max transformer w/ SoftQuantize |
| output levels | **bin index 0..3** | Eq. 1 |
| noise injection | `Q_in = Q_sim + eps`, `eps ~ N(0, sigma_noise)` | Eq. 6 |
| threshold parametrization | `Delta_i = ln(1+e^theta_i)`, `T_j = T_min + sum_k Delta_k` | Eq. 2, 3 |
| soft form | sigmoids at `k*(T_j - x)/tau_j` | Eq. 4 |
| local scale | `tau_j = (Delta_j + Delta_{j+1})/2` | Eq. 5 |
| temperature | `k_init ~ 1` annealed (cosine) to `k_max ~ 67` | Sec. IV A |
| stated cost of 2-bit | **5-10%** worse resolution than electron-level precision | Sec. VI |

**The mistake that made this section necessary.** The first bit-depth study placed thresholds by
occupancy on our own training charge and got `100 / 240 / 1295 e-`. T0 = 100 is 1.25 `sigma_noise`
above pedestal — it digitizes noise as signal — and the whole set sits a slot below the published
one. Those runs (`digi_*_lowfirst_midpoint_*`) answer a question nobody asked and are **not
comparable to the paper baselines**. Superseded, kept only as a "before".

Rules that follow:
- `level_mode='code'` is the apples-to-apples configuration. `'midpoint'` assumes a 4-entry
  code->charge LUT on-chip that the baselines do not have; any credit it earns was bought with chip
  area and must be labelled that way.
- No published thresholds exist for 3/4-bit. Use `scheme='lowfirst'`, which floors T0 at the
  published 248 e-, and say in the write-up that those are derived.
- Every run records its threshold provenance via `DigitizeLayer.describe()` into `summary.json`.
  If a constant is not published, the metadata must say so rather than silently substituting a
  heuristic.

Deliverables:
- A `DigitizeLayer` (configurable `n_bits`, thresholds, STE on/off) placed before `RouterFeatures`
  and the experts. **DONE** — `two_bit_optimization_helpers/symbolic/digitize.py`.
- A bit-depth ablation at the **published** thresholds, tracking **I68** (never residual std — it is
  tail-driven and produced a spurious non-monotonicity), Gaussian-fit pulls, and sign accuracy.
  **Superseded run pending re-run.**
- **Open — optimize *at* 2 bits:**
  1. Paper-threshold reference point, `scheme='paper'` + `level_mode='code'`, then the N=1 sweep.
  2. `code` vs `midpoint` at the published thresholds, both trained end-to-end.
  3. `TRAINABLE_THR=True`, initialised from the published set, `T_min=240`. Only **three** numbers
     exist at 2 bits and they are the entire ADC design; the deliverable is those three charge
     values plus how far they moved from the published ones.
  4. N=4 MoE at 2-bit. Only N=1 has ever been digitized; N=4 analog is 2.4x better on x.

### Goal 2 — Input noise robustness
Add a noise model at the input and train through it. End state: **a model that is robust under
2-bit input AND noise simultaneously.**
- `NOISE = -1` means **off** — verified, not assumed: `OptimizedDataGenerator_v3._read_tfrecord`
  guards the only noise site with `if self.noise != -1:`, and that site sits *above* the
  quantize/digitize branches. Otherwise `noise=[mu, sigma]`.
- Order is physical and fixed: the generator adds noise to the analog charge, then `DigitizeLayer`
  quantizes. Since the generator runs with `digitize=False`, the graph order is already correct.
- Grid: {analog, 2-bit} x sigma in {0, 40, **80** (nominal = `sigma_noise`), 160} e-, at the
  published thresholds.
- Also test train/test mismatch (train clean, test noisy) via `TEST_NOISE_SIGMA`, to separate
  "the model is robust" from "the model was trained on noise".
- **Report sign accuracy at every grid point.** The gate reads `Tx`, `Ty` — a small *differential*
  quantity, and the most noise-fragile thing in the model. Expect sign to degrade before the
  position resolutions do.

### Goal 3 — Knowledge distillation (LATER — do not start unprompted)
Blocked on the `labels_scale` audit in §5. Mentioned here only so it doesn't get architected away.
Keep the model/generator interfaces compatible with `PairedGen`.

## 7. Evaluation conventions (match the group's existing plots)

- Residuals: binned `sns.regplot` with sigma bands.
- Pulls: log-scale step histograms with a Gaussian fit overlaid. Target pull width ~1.0.
- Physical units: **um** for x/y, **degrees** for angles via `inverse_cot`.
- Always report the four outputs separately. x/y and cotA/cotB fail for different reasons.
- Every new run dumps a parquet with the **same schema** as `compare_baselines_vs_symbolic.ipynb`
  so comparisons stay apples-to-apples. New experiments go in a new output directory; never
  overwrite prior parquets.
- Compare against the paper baselines (Conv2D / Conv1D / MLP variants) already in `baseline_models/`.

## 8. People

- **Jennet** — PI. Flags architectural issues (coordinate origin offsets, the unsigned-angle problem).
- **Arghya (das214)** — built the ViT_Max teacher, `distill.py`, the MoE variants, and the
  input-digitization work being referenced here.
- **Brian** — new undergrad, onboarding. Any comments written into shared code should be short and
  general so they don't mislead him.
- **Harshul** — produced the convolved simulation datasets.

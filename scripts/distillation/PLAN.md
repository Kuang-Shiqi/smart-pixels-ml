# Symbolic + small-NN distillation from a teacher (ViT_Max)

Hybrid model that keeps a hand-readable physics formula in the loop. A
tiny NN absorbs hardware-side effects (Lorentz drift residuals, noise,
2-bit quantization artifacts, charge sharing nonlinearity). After
training, the teacher is discarded; deployment uses only
`symbolic + tiny_error_MLP`.

## Architecture

Three models. Train sequentially.

1. **Teacher** = `ViT_Max` (ported transformer, 418k params). Standard Max
   NLL on the 3sr centeredIncidence data via the existing repo pipeline
   (Part 1 soft-quantize + Part 2 2-bit). Frozen after training.

2. **Student means** = symbolic ansatz with a few trainable scalars.
   Three candidate forms swept in parallel against the same teacher:

   - **`barycenter`** (simplest): per-projection charge-weighted centroid
     for `x`, `y`; cluster-width estimator for `|cot a|`, `|cot b|`;
     Lorentz drift correction on `y` and `cot b`.
       - `x = sum_i q_i x_i / sum_i q_i`
       - `y = sum_i q_i y_i / sum_i q_i + dy/2`
       - `|cot a| = (w_x - 1) p_x / T`
       - `|cot b| = (w_y - 1) p_y / T  +/- T tan theta_L`
       - Trainable: `theta_L` (=> `dy = T tan theta_L`); per-output
         affine `(scale, bias)` for the angle outputs.
       - Fixed: `T = 100 um`, `p_x = 50 um`, `p_y = 12.5 um`.

   - **`localreco`** (arxiv 2602.15946 eq 8, 9): head-tail estimator
     using first/last pixel charges. Self-consistent: angles from
     `barycenter`'s cluster-width step feed into LocalReco's position
     formula.

   - **`blend`**: learned scalar mixes the above.
     `x = (1 - w) * x_bary + w * x_localreco`, w trainable per output.

   - **`pysr_aug`**: any variant above + PySR-discovered correction
     terms applied to the residual (`teacher_means - symbolic_means`),
     with their own trainable scalars.

3. **Student covariance** = tiny projected MLP. Architecture:
   - `AvgPool2D((1,16))` -> x-profile of length 16 (averaged over time + y)
   - `AvgPool2D((16,1))` -> y-profile of length 16 (averaged over time + x)
   - Concatenate -> 32-vector
   - `Dense(8, tanh)` -> `Dense(10, linear)` (10 Cholesky entries)
   - **~354 params total**. Same Cholesky parameterization as `custom_loss`
     in `loss.py`. Mirrors the repo's existing Conv1D-style x/y projection
     pattern and reuses physics-motivated projection.

   Total deployable footprint: ~6 symbolic scalars + 354 MLP params ~ 360
   params; ~5x smaller than the existing 1898-param Conv2D_Max baseline,
   ~1000x smaller than the ViT_Max teacher.

Final student distribution per event: `N(mu_S, Sigma_S)` where `mu_S`
comes from the symbolic head and `Sigma_S = L L^T` with `L`
constructed from the small MLP's 10 outputs.

## Sign of cot a / cot b

Cluster-width estimator gives `|cot|` only. Default: borrow sign from
teacher means, `sign(mu_teacher_cotA)` and `sign(mu_teacher_cotB)`.
After distillation, attach a 2-bit sign head to the small MLP so the
student is self-contained at deploy. (Two extra logits, negligible.)

## Loss

Per event:

```
L_total = L_data + lambda * C
L_data  = NLL_Max(y_true; mu_S, Sigma_S)        # student fits the data
C       = KL[ N(mu_T, Sigma_T) || N(mu_S, Sigma_S) ]   # student matches teacher
```

Both terms use the same per-event ground truth path; data NLL keeps the
student grounded in real labels even if the teacher is biased.

MDMM dual update on `lambda`:

```
lambda_{t+1} = max(0, lambda_t + eta * (C - c_target))
```

with `c_target = 0` (drive the KL to zero) and `eta` a small step size
(start `eta = 1e-3`). This is the standard Modified Differential Method
of Multipliers update: lambda grows when constraint is violated, shrinks
when it's slack. Implemented as a Keras callback that reads the
per-epoch mean KL and updates `lambda` as a `tf.Variable`.

Optional warm-up: `lambda` starts at 0 and ramps over the first N
epochs; pure data-NLL early so the small MLP can stabilize, then KL
kicks in.

## Training schedule

1. **Phase 0** (background, already running for baseline): Conv2D_Max
   under the standard Part 1 + Part 2 + eval. Used as a non-teacher
   baseline comparison.

2. **Phase 1**: train **ViT_Max teacher** under standard Max NLL via
   the existing `train_vit_max.py`. Queue to start after Phase 0
   finishes so the GPU isn't shared. Save best-val checkpoint.

3. **Phase 2** (optional): run **PySR** on `(teacher_means(x) -
   barycenter_means(x))` residuals over a held-out subset, fit
   closed-form correction terms. Promote the top expressions into a
   `pysr_aug` symbolic variant.

4. **Phase 3**: **distillation sweep**. For each symbolic variant in
   `{barycenter, localreco, blend, pysr_aug}`:
   - Build student = symbolic(variant) + small MLP (10 Chol).
   - Initialize MLP randomly, symbolic scalars at sensible defaults.
   - Train with `L_data + lambda * KL[teacher || student]` and MDMM
     dual update.
   - Track: data NLL, KL, lambda, total params, per-output residual
     mean and width on val.
   Pick the winner by lowest validation NLL subject to `mean KL <
   1e-2` (or similar threshold).

5. **Phase 4**: report. Compare winner to teacher and to the Conv2D_Max
   baseline. Tables: NLL, residual widths per output, parameter count,
   sym scalars learned. Plots: residual histograms overlay.

## Data and inputs

Same 3sr centeredIncidence dataset, same TFRecords already generated.
Teacher trained on the 2-bit digitized inputs (Part 2 style) so the
student sees the same input distribution at distillation time.

## Repo layout

```
smart-pixels-ml/
  two_bit_optimization_helpers/
    symbolic/
      __init__.py
      ansatz.py             # PhysicsAnsatz Keras layer, 3 variants
      sign.py               # teacher-borrow + sign-head helpers
    models/
      error_mlp.py          # tiny MLP -> 10 Chol entries
      student_max.py        # composes symbolic + error MLP -> Max output
    distill.py              # KL loss, MDMM callback, distill loop
    pysr_residuals.py       # PySR residual-discovery driver
  scripts/distillation/
    PLAN.md                 # this file
    train_teacher.py        # wraps train_vit_max.py launch
    distill_sweep.py        # runs the 4-variant sweep
```

## Open items to revisit after v1

- Sign-of-angle reliability (teacher-borrow vs sign-head MLP).
- Whether to retrain the teacher for longer if KL doesn't converge.
- Hardware-cost report: HLS / area / latency for the
  `symbolic + tiny_MLP` deploy module (the codesign path of the
  existing repo).

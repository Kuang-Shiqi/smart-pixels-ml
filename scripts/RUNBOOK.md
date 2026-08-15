# Runbook — training arms

**Training runs from `scripts/train_arm.py`. Plots run from the notebooks. Do not mix.**

Why: a notebook open in JupyterLab holds its own in-memory copy of the file. When it saves,
it overwrites whatever else wrote to that file. This has already cost us a fix — a threshold-
freeze patch written to `train_noise_ablation.ipynb` was silently reverted when JupyterLab
saved its stale copy, and the same race dropped a `NOISE_SIGMA` edit, which then produced a
run whose config did not match its output directory. Scripts have one writer.

## Environment

```
PY=/work/users/kuang14/smart_pixels/bin/python     # TF 2.15.1 / Keras 2.15 / qkeras
```

This is the interpreter behind the **SmartPixels** Jupyter kernel. The AF global pixi env
(`/work/pixi/global/...`) does **not** have `qkeras` and cannot run training.

Keras 2 matters: `add_weight` returns a `tf.Variable` whose `.trainable` is read-only, which
is why `threshold_deltas_raw.trainable = False` raises `AttributeError: can't set attribute`.
`freeze_adc_thresholds()` in the script handles it and verifies against
`model.trainable_variables`.

`/work` is **not visible to Slurm jobs**, so this env cannot be used on Hammer/Gilbreth. Batch
submission needs an env on `/depot` first — see "Not done yet" below.

## Run one arm

```
cd /depot/cms/private/users/kuang14/Smart_Pixel/smart-pixels-digitization
$PY scripts/train_arm.py --n-bits 2 --thr-scheme paper --level-mode code
```

`--dry-run` resolves the config, prints the plan and the output directory, and exits without
touching the GPU. Use it to check a command before committing an hour to it.

The run directory name encodes every knob that changes the answer, and the script **refuses to
overwrite an existing parquet** — pass `--overwrite` deliberately, or `--tag-suffix _v2` to
branch. Nothing silently replaces a prior result.

## The arms

| purpose | command |
|---|---|
| 2-bit reference (published T) | `--n-bits 2` |
| soft digitization, learned T | `--n-bits 2 --trainable-thr` |
| analog reference | `--n-bits none` |
| 3 / 4-bit reference | `--n-bits 3 --thr-scheme lowfirst` |
| LUT cost | `--n-bits 2 --level-mode midpoint` |
| noise sweep | `--n-bits 2 --noise-sigma {40,80,160}` |
| train clean / test noisy | `--n-bits 2 --noise-sigma 0 --test-noise-sigma 80` |
| train noisy / test clean | `--n-bits 2 --noise-sigma 80 --test-noise-sigma 0` |
| ADC under noise | `--n-bits 2 --trainable-thr --noise-sigma 80` |
| MoE | `--n-bits 2 --n-experts 4` |

**2 bits is the deployment point.** Analog and 3/4-bit are reference points that price what
quantization costs; no result is ever reported as "use N bits" for N != 2.

## Queue several

```
$PY scripts/train_arm.py --n-bits 2 --noise-sigma 40  > logs/noise40.log  2>&1
$PY scripts/train_arm.py --n-bits 2 --noise-sigma 160 > logs/noise160.log 2>&1
```

Run them **sequentially** — one A100, and two arms would both grow into it until the second
dies. Roughly 1 h per arm (P1 ~25 min, P2 ~26 min, P3 ~15 min).

Use `setsid nohup ... &` so a job survives a Jupyter kernel restart. It will **not** survive a
pod restart. A full-A100 session is culled after 24 h idle and a background process may not
count as activity, so keep queues under that and check in.

## Then plot

Nothing to edit. `compare_baselines_vs_symbolic.ipynb` sections B and C glob
`digi_*_nexp*` and pick up new runs automatically:

- **A** — axis fix, before vs after, in isolation
- **B** — digitization: thresholds published vs learned, I68 and sign vs bit depth
- **C** — noise: I68 and sign vs sigma, joined only within one quantizer

## Or, for a meeting

```
$PY_PLOT scripts/make_meeting_report.py --title "Aug 17 algorithm meeting update" \
                                        --slug 2026-08-17_algorithm_meeting
```
(`$PY_PLOT` = the AF global pixi env — no TF needed.) Writes
`reports/<slug>/` with `REPORT.md`, `summary.csv` and `plots/*.png` sized for slides.
Every number is recomputed from the parquets, and arms that have not finished are
skipped with a note rather than faked — so it is safe to run mid-queue and re-run after.

## Gotchas that already bit us

- **Never `--overwrite` to "redo" an arm** that a plot already quotes. New directory, always.
- `--thr-scheme paper` is 2-bit only; the paper publishes no 3/4-bit thresholds. The script
  rejects the combination rather than silently substituting a heuristic.
- The generator must not digitize (`digitize=False`, hard-coded): `DigitizeLayer` is the only
  quantizer. Two quantizers in series is silent and wrong.
- This dataset has **intrinsic negative charge** with the noise off (~2.9% of pixels, down to
  about -3400 e-). Any "negative means noise" logic is wrong here.
- The train and eval splits carry **different `labels_scale`** (0.42% on x, 0.64% on cotB).
  See `--label-scale-mode`; default `train` reproduces every run made so far.

## Not done yet

Batch submission to Hammer (`sbatch`, 13 GPU nodes, 30-day walltime, reachable from the AF
session). Blocked on building a TF+qkeras env on `/depot` — Slurm workers see neither `/work`
nor `/home`. `Env_Creation/environment.yaml` is the starting point. Worth doing when arms need
to run in parallel or overnight; not worth it while they are sequential and ~1 h.

---
name: train-arm
description: Train one arm of the SmartPixels symbolic-student study (bit depth, thresholds, noise, MoE) via scripts/train_arm.py, or queue several. Use whenever the user asks to run/launch/queue a training arm, an ablation point, a noise sweep, or a digitization configuration — e.g. "run the 2-bit noise 160 arm", "queue the mismatch arms", "train the MoE at 2 bits".
---

# Train one study arm

Training runs from `scripts/train_arm.py`, never from the notebooks. A notebook open in
JupyterLab overwrites the file when it saves, which has already reverted a fix and dropped a
config edit mid-study. Notebooks are for plots.

## Interpreter

```
PY=/work/users/kuang14/smart_pixels/bin/python      # TF 2.15.1 / Keras 2.15 / qkeras
```

The AF global pixi env has no qkeras and cannot train — but it *is* the right one for running
the plotting cells headlessly (`/work/pixi/global/.pixi/envs/default/bin/python`).

## Procedure

1. **Resolve the arm.** Map the request onto flags (table below). If it is ambiguous between two
   arms, ask — an arm is ~1 h of GPU.
2. **`--dry-run` first.** It prints the resolved config and the output directory without touching
   the GPU. Confirm the directory name matches the intent; the name encodes every knob.
3. **Check it does not already exist.** `ls -d /depot/cms/private/users/kuang14/Smart_Pixel/digi_*`.
   The script refuses to overwrite an existing parquet. Never pass `--overwrite` to "redo" an arm
   a plot already quotes — use `--tag-suffix _v2` and keep both.
4. **Launch.** One arm: run it in the background and report the log path. Several: append to
   `scripts/run_queue.sh` and `setsid nohup bash scripts/run_queue.sh > logs/queue.log 2>&1 &`.
   **Sequential only** — one A100 MIG 1g.5gb slice (5 GB); parallel arms OOM.
5. **Report** when it lands: I68 (never residual std), sign accuracy for both angles, pull sigma
   (~1.0 = calibrated), and for trainable-threshold arms the learned T and its offset from the
   published 248/668/1663 e-.
6. **Re-run the plots**: sections B and C of `compare_baselines_vs_symbolic.ipynb` glob
   `digi_*_nexp*` and pick up new runs with no edits.

## Arms

| purpose | flags |
|---|---|
| 2-bit reference (published T) | `--n-bits 2` |
| soft digitization, learned T | `--n-bits 2 --trainable-thr` |
| analog reference | `--n-bits none` |
| 3 / 4-bit reference | `--n-bits 3 --thr-scheme lowfirst` |
| LUT cost | `--n-bits 2 --level-mode midpoint` |
| noise | `--n-bits 2 --noise-sigma {40,80,160}` |
| train clean / test noisy | `--n-bits 2 --noise-sigma 0 --test-noise-sigma 80` |
| ADC under noise | `--n-bits 2 --trainable-thr --noise-sigma 80` |
| MoE | `--n-bits 2 --n-experts 4` |

**2 bits is the deployment point** — the on-sensor ADC is 2-bit. Analog and 3/4-bit are reference
points that price what quantization costs. Never report a result as "use N bits" for N != 2.
`--thr-scheme paper` is 2-bit only; the paper publishes no 3/4-bit thresholds and the script
rejects the combination rather than substituting a heuristic.

## Watch for

- **Session lifetime.** Jobs die with the pod. Keep a queue well under the idle-cull window and
  check in; `setsid` protects against a kernel restart only.
- **Sign accuracy is the fragile number** under noise — the gate reads a between-slice centroid
  *drift*, so it degrades before the position resolutions do. Report it at every grid point.
- **Pre-train asserts are load-bearing.** Axis probe (x-slot must read +375 µm), init sign
  agreement against the arm's own measured gate ceiling, NaN scan. If one trips, do not weaken it
  — find out why. A trip usually means stale state, not bad physics.

Full detail, including the environment and the known gotchas: `scripts/RUNBOOK.md`.

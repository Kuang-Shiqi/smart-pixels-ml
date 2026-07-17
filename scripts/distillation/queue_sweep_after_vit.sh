#!/bin/bash
# Poll until train_vit_max.py exits, then launch the distillation sweep
# with the ViT_Max teacher. Reads checkpoint path from
# runs/vit_max_run/summary.json.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
LOG=$REPO/runs/distill_sweep_vit_teacher/queue.log
SUMMARY=$REPO/runs/vit_max_run/summary.json
THRESHOLDS=$REPO/runs/vit_max_run/optimized_thresholds.json
mkdir -p "$(dirname "$LOG")"
{
  echo "[$(date '+%H:%M:%S')] watcher pid=$$ waiting on train_vit_max.py"
  until ! pgrep -f "train_vit_max.py" >/dev/null; do
    sleep 120
  done
  echo "[$(date '+%H:%M:%S')] train_vit_max.py done; checking artifacts"
  if [[ ! -f "$SUMMARY" ]] || [[ ! -f "$THRESHOLDS" ]]; then
    echo "[$(date '+%H:%M:%S')] missing summary.json or thresholds; aborting sweep launch"
    exit 1
  fi
  CKPT=$(/work/users/das214/envs/smartpix-2bit/bin/python -c \
    "import json; print(json.load(open('$SUMMARY'))['part2_checkpoints'])")
  echo "[$(date '+%H:%M:%S')] teacher_checkpoints=$CKPT"
  cd "$REPO"
  /work/users/das214/envs/smartpix-2bit/bin/python -u \
    scripts/distillation/distill_sweep.py \
    --teacher-checkpoints "$CKPT" \
    --teacher-model-type ViT_Max \
    --thresholds-json "$THRESHOLDS" \
    --epochs 30 \
    --warmup-steps 200 \
    --out runs/distill_sweep_vit_teacher
  echo "[$(date '+%H:%M:%S')] sweep finished, rc=$?"
} >> "$LOG" 2>&1

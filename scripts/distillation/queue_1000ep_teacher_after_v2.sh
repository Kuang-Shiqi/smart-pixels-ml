#!/bin/bash
# Wait for the v2 distill sweep to exit, then launch train_vit_max_1000.py
# (1000+1000-epoch ViT_Max teacher) under the same fully-detached pattern.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
LOG=$REPO/runs/vit_max_run_1000ep/queue.log
mkdir -p "$(dirname "$LOG")"
{
  echo "[$(date '+%H:%M:%S')] watcher pid=$$ waiting on v2 distill_sweep.py"
  until ! pgrep -f "distill_sweep.py.*distill_sweep_vit_teacher_v2" >/dev/null; do
    sleep 120
  done
  echo "[$(date '+%H:%M:%S')] v2 sweep done; launching 1000-epoch ViT_Max teacher"
  cd "$REPO"
  nohup setsid /work/users/das214/envs/smartpix-2bit/bin/python -u \
    train_vit_max_1000.py \
    > runs/vit_max_run_1000ep/train.log 2>&1 < /dev/null &
  disown
  echo "[$(date '+%H:%M:%S')] launched, pid=$!  expected wallclock ~30h"
} >> "$LOG" 2>&1

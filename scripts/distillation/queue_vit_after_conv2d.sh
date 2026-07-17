#!/bin/bash
# Poll until train_conv2d_max.py exits, then launch train_vit_max.py
# (the ViT_Max teacher) under the same detached pattern.
set -u
LOG=/work/users/das214/SmartPixels/smart-pixels-ml/runs/vit_max_run/queue.log
mkdir -p "$(dirname "$LOG")"
{
  echo "[$(date '+%H:%M:%S')] watcher pid=$$ waiting on train_conv2d_max.py"
  until ! pgrep -f "train_conv2d_max.py" >/dev/null; do
    sleep 60
  done
  echo "[$(date '+%H:%M:%S')] Conv2D_Max done; launching ViT_Max teacher"
  cd /work/users/das214/SmartPixels/smart-pixels-ml
  nohup setsid /work/users/das214/envs/smartpix-2bit/bin/python -u train_vit_max.py \
    > runs/vit_max_run/train.log 2>&1 < /dev/null &
  disown
  echo "[$(date '+%H:%M:%S')] launched, vit pid=$!"
} >> "$LOG" 2>&1

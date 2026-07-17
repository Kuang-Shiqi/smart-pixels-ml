#!/bin/bash
# Chained watcher: wait for train_vit_max_1000.py to finish, then
#   1. Launch the 1000-epoch distillation sweep (3 variants) against the
#      new ViT_Max teacher.
#   2. Wait for the sweep to finish, then generate plots + final REPORT.md.
set -u
REPO=/work/users/das214/SmartPixels/smart-pixels-ml
PY=/work/users/das214/envs/smartpix-2bit/bin/python
SWEEP_DIR=$REPO/runs/distill_sweep_vit_teacher_1000ep
TEACHER_DIR=$REPO/runs/vit_max_run_1000ep
LOG=$SWEEP_DIR/queue.log
mkdir -p "$SWEEP_DIR"
{
  echo "[$(date '+%H:%M:%S')] watcher pid=$$ waiting on train_vit_max_1000.py (this takes ~30h)"
  until ! pgrep -f "train_vit_max_1000.py" >/dev/null; do
    sleep 300
  done
  echo "[$(date '+%H:%M:%S')] 1000-ep teacher finished. checking artifacts."
  if [[ ! -f "$TEACHER_DIR/summary.json" ]] || [[ ! -f "$TEACHER_DIR/optimized_thresholds.json" ]]; then
    echo "[$(date '+%H:%M:%S')] missing teacher summary / thresholds. ABORT."
    exit 1
  fi
  CKPT=$($PY -c "import json; print(json.load(open('$TEACHER_DIR/summary.json'))['part2_checkpoints'])")
  echo "[$(date '+%H:%M:%S')] teacher ckpt=$CKPT"

  echo "[$(date '+%H:%M:%S')] launching 1000-ep distillation sweep (3 variants)"
  cd "$REPO"
  $PY -u scripts/distillation/distill_sweep.py \
    --teacher-checkpoints "$CKPT" \
    --teacher-model-type ViT_Max \
    --thresholds-json "$TEACHER_DIR/optimized_thresholds.json" \
    --epochs 1000 \
    --warmup-steps 200 \
    --out "$SWEEP_DIR"
  RC=$?
  echo "[$(date '+%H:%M:%S')] sweep finished, rc=$RC"

  if [[ $RC -eq 0 ]]; then
    echo "[$(date '+%H:%M:%S')] saving student parquets"
    $PY scripts/distillation/save_student_parquets.py \
      --sweep-dir "$SWEEP_DIR" \
      --teacher-summary "$TEACHER_DIR/summary.json" \
      --thresholds-json "$TEACHER_DIR/optimized_thresholds.json" \
      --out-dir "$REPO/runs/processed_parquets/test_3src/2bit_optimized_students_1000ep"
    echo "[$(date '+%H:%M:%S')] generating plots + final report"
    $PY scripts/distillation/report_final_1000ep.py
  else
    echo "[$(date '+%H:%M:%S')] sweep failed; skipping report"
  fi
} >> "$LOG" 2>&1

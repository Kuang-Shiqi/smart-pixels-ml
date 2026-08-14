#!/bin/bash
# Run the remaining study arms, one at a time.
#
# SEQUENTIAL on purpose: one A100 MIG slice (5 GB). Two arms would both grow into it
# until the second dies. ~1 h per arm.
#
#   setsid nohup bash scripts/run_queue.sh > logs/queue.log 2>&1 &
#
# setsid detaches it from the Jupyter kernel, so restarting a kernel does not kill it.
# It does NOT survive a pod restart.
set -u
cd "$(dirname "$0")/.."
PY=/work/users/kuang14/smart_pixels/bin/python
mkdir -p logs

run () {                      # run <logname> <args...>
  local name="$1"; shift
  if [ -f "logs/${name}.done" ]; then echo "SKIP  ${name} (already done)"; return 0; fi
  echo "===== START ${name}  $(date '+%F %T')"
  if "$PY" scripts/train_arm.py "$@" > "logs/${name}.log" 2>&1; then
    touch "logs/${name}.done"
    echo "===== OK    ${name}  $(date '+%F %T')"
    grep -E "^(x|y|cotA|cotB) |sign accuracy|learned T|moved from init|vs published" \
         "logs/${name}.log" | sed 's/^/      /'
  else
    echo "===== FAIL  ${name}  $(date '+%F %T')  (exit $?) -- see logs/${name}.log"
    tail -15 "logs/${name}.log" | sed 's/^/      /'
  fi
}

# --- noise sweep: brackets the nominal 80 e- point already on disk ---
run noise40   --n-bits 2 --noise-sigma 40
run noise160  --n-bits 2 --noise-sigma 160

# --- train/test mismatch: robust model vs model trained on noise ---
run test80    --n-bits 2 --noise-sigma 0  --test-noise-sigma 80
run noise80_test0 --n-bits 2 --noise-sigma 80 --test-noise-sigma 0

# --- does the optimal ADC move when the input is noisy? ---
run trainthr_noise80 --n-bits 2 --trainable-thr --noise-sigma 80

# --- MoE at the deployment point ---
run nexp4     --n-bits 2 --n-experts 4

echo "QUEUE COMPLETE $(date '+%F %T')"

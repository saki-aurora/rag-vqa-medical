#!/usr/bin/env bash
set -euo pipefail

REPO="/root/work/rag-vqa-medical"
PY="/root/work/venv-vqa/bin/python"
LIMUC_ROOT="$REPO/Prototyping_reformat/DatasetAnalysis/LIMUC"
SCRIPT="$REPO/Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/17_pass6_generative_multiseed.py"
OUT_LOG="$REPO/Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_objfix_b200.driver.log"
WATCH_LOG="$REPO/Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/auto_switch_after_seed011.log"
SEED011_METRICS="$REPO/Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_lora_objfix_b200_seed011/metrics_test.json"

log() {
  echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*" | tee -a "$WATCH_LOG"
}

cd "$REPO"
log "watcher started"

while [[ ! -f "$SEED011_METRICS" ]]; do
  log "waiting: seed011 not finished yet"
  sleep 30
done
log "detected seed011 completion (metrics_test.json exists)"

DRIVER_PIDS=$(pgrep -f "17_pass6_generative_multiseed.py" || true)
if [[ -n "$DRIVER_PIDS" ]]; then
  log "stopping existing driver pids: $DRIVER_PIDS"
  kill $DRIVER_PIDS || true
  sleep 5
  REMAIN=$(pgrep -f "17_pass6_generative_multiseed.py" || true)
  if [[ -n "$REMAIN" ]]; then
    log "forcing stop for remaining driver pids: $REMAIN"
    kill -9 $REMAIN || true
  fi
else
  log "no active multiseed driver found"
fi

OTHER_PIDS=$(pgrep -f "train_vlm_lora_mayo.py .*vlm_lora_objfix_b200_seed0(23|42)" || true)
if [[ -n "$OTHER_PIDS" ]]; then
  log "stopping lingering seed023/042 trainers: $OTHER_PIDS"
  kill $OTHER_PIDS || true
  sleep 3
  OTHER_REMAIN=$(pgrep -f "train_vlm_lora_mayo.py .*vlm_lora_objfix_b200_seed0(23|42)" || true)
  if [[ -n "$OTHER_REMAIN" ]]; then
    kill -9 $OTHER_REMAIN || true
  fi
fi

log "relaunching driver for seeds 023/042 with num-workers=8"
nohup "$PY" "$SCRIPT" \
  --limuc-root "$LIMUC_ROOT" \
  --python "$PY" \
  --tag pass6_generative_objfix_b200 \
  --new-seeds 23,42 \
  --existing-runs vlm_lora_objfix_b200_seed011 \
  --run-prefix vlm_lora_objfix_b200_seed \
  --epochs 2 --batch-size 2 --grad-accum 4 --lr 5e-5 --weight-decay 0.0 \
  --max-new-tokens 8 --num-workers 8 --logging-steps 25 --save-steps 400 --save-total-limit 1 \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
  --balanced-sampling --force-cuda \
  --label-token-only --class-token-loss-weight 1.0 --template-token-loss-weight 0.0 \
  --eval-mode2-strategy sequence_logprob \
  --exclude-nonconverged-mode1 --force-retrain \
  > "$OUT_LOG" 2>&1 &

sleep 2
NEW_PID=$(pgrep -fo "17_pass6_generative_multiseed.py .*--new-seeds 23,42" || true)
if [[ -n "$NEW_PID" ]]; then
  log "relaunch successful: driver pid=$NEW_PID"
else
  log "warning: relaunch PID not detected yet; check $OUT_LOG"
fi

log "watcher completed"

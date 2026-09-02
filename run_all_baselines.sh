#!/usr/bin/env bash
# [SAFETAIL][FIX][D-14] Run the baseline sweep.
#
# Was: PYTHON hardcoded to /home/jyoti/miniconda3/bin/python; comments claimed
# CSVs land in src/results/ then src/logs/ -- neither is true. They land in
# ./<TRAINING_LOG_FOLDER>/ (latency_log.csv, episode_rewards.csv, ...).
#
# Now: PYTHON defaults to the repo venv, overridable via $SAFETAIL_PY. K ranges
# 1..BETA (B6). Set SEEDS / KS to control the matrix. Output dirs follow the
# plan.md 9.1 convention  results/<mode>_k<K>_s<SEED>/.
#
# Usage:
#   bash run_all_baselines.sh                       # families x K{1..5} x SEEDS
#   SEEDS="0 1 2" KS="1 3 5" bash run_all_baselines.sh
#   SAFETAIL_PY=python bash run_all_baselines.sh
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO" || exit 2
export PYTHONUTF8=1 KMP_DUPLICATE_LIB_OK=TRUE

PY="${SAFETAIL_PY:-$REPO/.venv/Scripts/python.exe}"
[ -x "$PY" ] || PY="python"
SCRIPT="src/main.py"
FAMILIES="${FAMILIES:-minload minprop rand}"
KS="${KS:-1 2 3 4 5}"
SEEDS="${SEEDS:-0 1 2}"
GITSHA="$(git rev-parse --short=7 HEAD 2>/dev/null || echo nogit)"

port=6001
mkdir -p results logs
echo "python: $PY   git: $GITSHA   families: $FAMILIES   K: $KS   seeds: $SEEDS"

for fam in $FAMILIES; do
  for k in $KS; do
    for seed in $SEEDS; do
      mode="${fam}_${k}"
      run="${mode}_s${seed}_${GITSHA}"
      outdir="results/${run}"
      mkdir -p "$outdir"
      echo "-> $run  (port $port)"
      BASELINE_MODE="$mode" RECEIVER_PORT="$port" SAFETAIL_SEED="$seed" \
        TRAINING_LOG_FOLDER="$outdir" \
        nohup "$PY" "$SCRIPT" >> "logs/${run}.txt" 2>&1 &
      echo "   pid $!  logs/${run}.txt"
      port=$((port + 1))
    done
  done
done

echo ""
echo "All runs launched. Results in results/<mode>_k<K>_s<seed>_<sha>/."
echo "Monitor:  tail -f logs/<run>.txt        PIDs:  ps | grep main.py"
echo "NOTE: full runs are long. Use SAFETAIL_SMOKE=1 for a plumbing check."

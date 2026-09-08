#!/usr/bin/env bash
# Run the temp-0 vs sampled judge experiment end to end.
#
#   cd /workspace/research && nohup bash run_unattended.sh > /workspace/run.log 2>&1 &
#
# This script NEVER stops the pod. Pod lifecycle is the operator's responsibility.
#
# It does:
#   * abort before the expensive arm if the determinism gate fails
#   * write everything to /workspace (network volume) so results survive a restart
#   * leave the pod running on every exit path so results can be read afterwards

set -uo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
PY="${PY:-/workspace/venv/bin/python}"
OUT=/workspace/results
GATE="${GATE:-0.99}"        # decision-level stability, not bit-identity

export HF_HOME=/workspace/hf
export PATH=/workspace/venv/bin:$PATH   # vLLM's engine needs ninja on PATH
mkdir -p "$OUT"
cd /workspace/research

finish() {
  echo ""
  echo "=== EXIT $? at $(date -u) ==="
  cp -r /workspace/research/results/* "$OUT"/ 2>/dev/null
  sync
  echo "results in $OUT"
  echo "POD LEFT RUNNING - stop it yourself when done."
}
trap finish EXIT

echo "=== $(date -u) model=$MODEL ==="

echo ""
echo "=== STEP 1/3  determinism gate (T=0, 2 shuffled draws) ==="
$PY scripts/15_run_judge.py --backend vllm --model "$MODEL" --temp 0 --draws 2 --batch 32 2>&1 \
  | grep -vE "Processed prompts|Rendering prompts|Capturing CUDA" || { echo "T0 RUN FAILED"; exit 1; }
$PY scripts/16_analyze_judge.py --runs T0.0_k2 --fpr 0.05 --families real || { echo "T0 ANALYSIS FAILED"; exit 1; }

DET=$($PY -c "import json;print(json.load(open('results/judge_analysis.json')).get('T0.0_k2',{}).get('determinism','nan'))")
echo "DETERMINISM=$DET  (gate >= $GATE)"
$PY -c "import sys;v=float('$DET');sys.exit(0 if v>=$GATE else 1)" || {
  echo "GATE FAILED: temp-0 decisions are not stable enough ($DET)."
  echo "Not spending the sampled arm. Pod left running for diagnosis."
  exit 2; }
echo "GATE PASSED"

echo ""
echo "=== STEP 2/3  sampled arm (T=1.0, 5 draws) ==="
$PY scripts/15_run_judge.py --backend vllm --model "$MODEL" --temp 1.0 --draws 5 --batch 32 2>&1 \
  | grep -vE "Processed prompts|Rendering prompts|Capturing CUDA" || { echo "SAMPLED RUN FAILED (resumable)"; exit 1; }

echo ""
echo "=== STEP 3/3  final analysis ==="
$PY scripts/16_analyze_judge.py --runs T0.0_k2 T1.0_k5 --fpr 0.05 --families real
echo ""
echo "=== DONE $(date -u) ==="

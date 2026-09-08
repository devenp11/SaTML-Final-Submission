#!/usr/bin/env bash
# Closes the one open weakness in the paper's negative result (Section 7):
# it rested on a single judge at a single temperature.
#
#   ARM A  second judge model, T=0 and T=1.0  -> is the null a Qwen-1.5B artifact?
#   ARM B  dose-response on the original judge, T=0.3 / 0.7 / 1.0
#
# Both use --score-mode sample: a 0.5 threshold on the two-way softmax ratio is exactly
# temperature-invariant and cannot produce a sampled arm (Section 7's recorded pitfall).
#
# --tag is REQUIRED for the second judge. Output filenames key on temperature and draws
# only, so without it the 7B run would resume from and overwrite the 1.5B scores.
#
# This script NEVER stops the pod. Stop it yourself when the run is done.
#
#   nohup bash scripts/26_judge_robustness.sh > /workspace/robust.log 2>&1 &

set -uo pipefail
cd /workspace/research
export HF_HOME=/workspace/hf
export PATH=/workspace/venv/bin:$PATH        # vLLM's engine subprocess needs ninja
PY=/workspace/venv/bin/python

J1="${J1:-Qwen/Qwen2.5-1.5B-Instruct}"       # original judge, already cached
J2="${J2:-Qwen/Qwen2.5-7B-Instruct}"         # second judge, cached from the first session
TAG2=q7b
BATCH=32
QUIET='Processed prompts|Rendering prompts|Capturing CUDA|it/s\]'

finish() {
  echo ""
  echo "=== EXIT $? at $(date -u) ==="
  mkdir -p /workspace/results
  cp -r /workspace/research/results/* /workspace/results/ 2>/dev/null
  sync
  echo "results copied to /workspace/results (survives pod stop)"
  echo "POD LEFT RUNNING - stop it yourself."
}
trap finish EXIT

run() {   # run <model> <label> <temp> <draws> [tag]
  echo ""
  echo "--- $2   model=$1  T=$3  draws=$4  $(date -u) ---"
  $PY scripts/15_run_judge.py --backend vllm --model "$1" \
      --temp "$3" --draws "$4" --batch $BATCH --score-mode sample \
      ${5:+--tag "$5"} 2>&1 | grep -vE "$QUIET"
}

echo "############ ARM A: second judge ($J2) ############"
run "$J2" A-greedy  0    2 "$TAG2"
run "$J2" A-sampled 1.0  5 "$TAG2"

echo ""
echo "############ ARM B: dose-response on $J1 ############"
run "$J1" B-t03 0.3 5
run "$J1" B-t07 0.7 5
# T=0 k2 and T=1.0 k5 for J1 already exist; 15_run_judge resumes and re-scores nothing.

echo ""
echo "############ ANALYSIS ############"
echo ""
echo "== A. second judge, 0.5 floor (comparable to the paper's Table) =="
$PY scripts/16_analyze_judge.py --runs T0.0_k2_s_$TAG2 T1.0_k5_s_$TAG2 \
    --fpr 0.05 --families real

echo ""
echo "== A. second judge, strict FPR-matched threshold (no floor) =="
echo "   If this reports 0 families, the judge separates benign from attacks so cleanly"
echo "   that the 5% FPR quantile sits near zero and nothing evades -- that IS the"
echo "   'FPR constraint is slack' limitation, shown rather than asserted."
$PY scripts/16_analyze_judge.py --runs T0.0_k2_s_$TAG2 T1.0_k5_s_$TAG2 \
    --fpr 0.05 --min-thr 0.0 --families real

echo ""
echo "== B. dose-response, original judge =="
$PY scripts/16_analyze_judge.py --runs T0.0_k2_s T0.3_k5_s T0.7_k5_s T1.0_k5_s \
    --fpr 0.05 --families real

echo ""
echo "=== ROBUSTNESS_DONE $(date -u) ==="

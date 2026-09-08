"""Run the judge over the unique payloads at a given temperature, k draws.

  # CPU dry run, no model, no GPU:
  python scripts/15_run_judge.py --backend stub --temp 0   --draws 1 --limit 400
  python scripts/15_run_judge.py --backend stub --temp 1.0 --draws 5 --limit 400

  # timing probe on the pod (do this FIRST, extrapolate before the full run):
  python scripts/15_run_judge.py --backend vllm --model meta-llama/Llama-3.1-8B-Instruct \
      --temp 0 --draws 1 --limit 200

  # real runs:
  python scripts/15_run_judge.py --backend vllm --model ... --temp 0   --draws 1
  python scripts/15_run_judge.py --backend vllm --model ... --temp 1.0 --draws 5

Writes results/judge_scores_T{temp}_k{draws}.jsonl : one row per (payload_id, draw).
Resumable: an existing output file is read and completed draws are skipped.
"""
import sys, json, argparse, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.judge import get_backend
from config import DATA_DIR, RESULTS_DIR, JUDGE_RAW_DIR

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="stub", choices=["stub", "vllm"])
ap.add_argument("--model", default=None)
ap.add_argument("--temp", type=float, required=True)
ap.add_argument("--draws", type=int, default=1)
ap.add_argument("--limit", type=int, default=0, help="cap unique payloads (dry run / timing probe)")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--max-model-len", type=int, default=16384)
ap.add_argument("--seed0", type=int, default=1000)
ap.add_argument("--tag", default="",
                help=("suffix for the output filename. REQUIRED when running a different "
                      "judge model: filenames key on temperature and draws only, so a "
                      "second model would resume from and overwrite the first model's "
                      "scores, silently mixing two judges in one file."))
ap.add_argument("--score-mode", choices=["logprob", "sample"], default="logprob",
                help=("logprob = P(YES) ratio from first-token logprobs; T-INVARIANT at a 0.5 "
                      "threshold, so it CANNOT produce a sampled arm. sample = the judge's "
                      "actual emitted verdict, genuinely stochastic in temperature."))
a = ap.parse_args()

if a.temp == 0 and a.draws > 1:
    print("temp=0 with draws>1: this IS the determinism check. Payload order is shuffled "
          "per draw so batch composition differs between passes -- that is what surfaces "
          "vLLM's batch-dependent nondeterminism. Expect det=1.000 in script 16.")

# unique payloads, in stable id order
payloads, benign_ids = {}, set()
with open(DATA_DIR / "judge_payloads.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        payloads.setdefault(r["payload_id"], r["text"])
        if r["kind"] == "benign":
            benign_ids.add(r["payload_id"])
ids = sorted(payloads)
if a.limit:
    # The benign set carries the FPR calibration, and its ids sort last, so a naive
    # head-truncation yields a run that cannot be calibrated. Always keep benign.
    ids = sorted(set(ids[:a.limit]) | benign_ids)
    print(f"  (limit={a.limit}; {len(benign_ids)} benign payloads force-included for calibration)")
print(f"unique payloads to judge: {len(ids):,}   temp={a.temp}  draws={a.draws}  backend={a.backend}")

JUDGE_RAW_DIR.mkdir(parents=True, exist_ok=True)
mode_tag = "" if a.score_mode == "logprob" else "_s"
model_tag = ("_" + a.tag) if a.tag else ""
# written alongside the shipped raw outputs, so a GPU re-run lands where the
# analysis scripts already look
out_path = JUDGE_RAW_DIR / f"judge_scores_T{a.temp}_k{a.draws}{mode_tag}{model_tag}.jsonl"
print(f"output: {out_path.name}")

done = set()
if out_path.exists():
    with open(out_path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                done.add((r["payload_id"], r["draw"]))
            except json.JSONDecodeError:
                continue
    print(f"resuming: {len(done):,} (payload,draw) already scored")

if a.temp > 0 and a.score_mode == "logprob":
    print("WARNING: temp>0 with --score-mode logprob is a NO-OP. A 0.5 threshold on the "
          "two-way softmax ratio is invariant to temperature, so this cannot produce a "
          "sampled arm. Use --score-mode sample.")
be = get_backend(a.backend, a.model,
                 **({"max_model_len": a.max_model_len, "score_mode": a.score_mode}
                    if a.backend == "vllm" else {}))

t0 = time.time()
n_new = n_tok = 0
with open(out_path, "a", encoding="utf-8") as out:
    for draw in range(a.draws):
        todo = [i for i in ids if (i, draw) not in done]
        if not todo:
            continue
        seed = a.seed0 + draw
        # Shuffle order per draw so batch composition differs between passes.
        # At temp 0 this is what makes the determinism check meaningful: identical
        # scores under different batching means genuinely deterministic inference.
        import random
        random.Random(seed).shuffle(todo)
        for s in range(0, len(todo), a.batch):
            chunk = todo[s:s + a.batch]
            texts = [payloads[i] for i in chunk]
            scores = be.score_batch(texts, a.temp, seed)
            for i, sc in zip(chunk, scores):
                out.write(json.dumps({"payload_id": i, "draw": draw, "temp": a.temp,
                                      "seed": seed, "score": float(sc)}) + "\n")
            out.flush()
            n_new += len(chunk)
            n_tok += sum(len(t) for t in texts) / 4.0
            el = time.time() - t0
            if (s // a.batch) % 10 == 0 or s + a.batch >= len(todo):
                rate = n_new / el if el > 0 else 0
                tps = n_tok / el if el > 0 else 0
                rem = (len(todo) - s - len(chunk)) + (a.draws - draw - 1) * len(ids)
                eta = rem / rate / 3600 if rate > 0 else float("nan")
                print(f"  draw {draw} {s+len(chunk):>6,}/{len(todo):,}  "
                      f"{rate:6.1f} payloads/s  ~{tps:8,.0f} prefill tok/s  ETA {eta:5.2f} h")

el = time.time() - t0
print(f"\nscored {n_new:,} new (payload,draw) in {el/60:.1f} min")
if n_new:
    print(f"throughput: {n_new/el:.1f} payloads/s, ~{n_tok/el:,.0f} prefill tokens/s")
    full = len(payloads) * a.draws
    print(f"EXTRAPOLATION: full run of {full:,} calls at this rate = {full/(n_new/el)/3600:.2f} hours")
print(f"wrote {out_path}")

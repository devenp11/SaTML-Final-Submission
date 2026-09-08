"""Size the temp-0 vs sampled judge experiment before renting anything.

Counts the payloads that actually need judging and their length distribution,
then estimates total prefill tokens under a few sampling budgets.
"""
import sys, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import parse_scenario, OBJ_KEYS
from config import DATA_DIR, PHASE1_LEVELS, RESULTS_DIR


def tokens(s):
    return set(re.findall(r"\w+", (s or "").lower()))


# Payloads we need: for every (team, defense) cell with a success, the first
# successful payload plus all its near-variants (Jaccard >= 0.7) and exact dups.
cells = defaultdict(lambda: defaultdict(list))
with open(DATA_DIR / "phase1_succteams_full.jsonl", encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        _, var = parse_scenario(r.get("scenario"))
        if var not in PHASE1_LEVELS:
            continue
        obj = r.get("objectives")
        obj = json.loads(obj) if isinstance(obj, str) else (obj or {})
        e2e = all(bool(obj.get(k, False)) for k in OBJ_KEYS)
        txt = ((r.get("subject") or "") + "\n" + (r.get("body") or "")).strip()
        cells[PHASE1_LEVELS[var]][r.get("team_id")].append((r.get("started_time") or "", e2e, txt))

need = set()
n_fam = 0
for key, teams in cells.items():
    for team, subs in teams.items():
        subs.sort(key=lambda x: x[0])
        fi = next((i for i, (_, e, _) in enumerate(subs) if e), None)
        if fi is None:
            continue
        n_fam += 1
        S = tokens(subs[fi][2])
        need.add(subs[fi][2])
        for _, e, txt in subs[fi + 1:]:
            T = tokens(txt)
            j = len(S & T) / len(S | T) if (S | T) else 1.0
            if j >= 0.7:
                need.add(txt)

lens = np.array([len(t) for t in need])
# ~4 chars/token is the standard rough ratio for English + code-ish text
tok = lens / 4.0

fp = json.loads((Path(__file__).resolve().parent.parent / "data" / "emails_for_fp_tests.json").read_text(encoding="utf-8"))
fp_texts = fp if isinstance(fp, list) else list(fp.values())
fp_lens = np.array([len(json.dumps(x)) for x in fp_texts])

print(f"exploit families (first-success payloads) : {n_fam:,}")
print(f"UNIQUE payloads needing judgement         : {len(need):,}")
print(f"benign emails for FPR calibration         : {len(fp_texts):,}")
print()
print(f"payload chars   median={np.median(lens):,.0f}  mean={lens.mean():,.0f}  p90={np.percentile(lens,90):,.0f}  max={lens.max():,.0f}")
print(f"payload ~tokens median={np.median(tok):,.0f}  mean={tok.mean():,.0f}  p90={np.percentile(tok,90):,.0f}  max={tok.max():,.0f}")
print(f"benign  chars   median={np.median(fp_lens):,.0f}  p90={np.percentile(fp_lens,90):,.0f}")
print()

N = len(need) + len(fp_texts)
JUDGE_PROMPT_TOK = 250          # system + instructions wrapper
for k in (1, 5, 8, 16):
    calls = N * (1 + k)         # 1 greedy pass + k sampled passes
    prefill = calls * (tok.mean() + JUDGE_PROMPT_TOK)
    print(f"k={k:2d} sampled draws/payload -> {calls:9,.0f} judge calls, ~{prefill/1e6:7.1f}M prefill tokens")

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "judge_workload_size.json").write_text(json.dumps({
    "n_families": n_fam, "n_unique_payloads": len(need), "n_benign": len(fp_texts),
    "median_tokens": float(np.median(tok)), "mean_tokens": float(tok.mean()),
    "p90_tokens": float(np.percentile(tok, 90)), "max_tokens": float(tok.max()),
}, indent=1), encoding="utf-8")

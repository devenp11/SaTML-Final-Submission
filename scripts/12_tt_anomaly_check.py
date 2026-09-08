"""Why does Tensor Trust hazard RISE from attempt 1 to attempts 2-3, unlike LLMail?

H1 PROBING: attempt 1 is often a short guess at the access code, not a real attack.
   -> compare attempt-1 text length / content to attempts 2-3.
H2 ARTIFACT: cells are conditioned on >=2 attempts, so attempt 1 of a cell that
   succeeded immediately (1 attempt) is excluded -> depresses bin 1.
   -> recompute hazard including ALL cells (min_attempts=1).
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tensortrust import load_cells
from config import RESULTS_DIR

BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 10**9)]


def hazard(cs):
    rows = []
    for lo, hi in BINS:
        risk = ev = 0
        for v in cs.values():
            flags = [s for _, s, _, _ in v]
            first = next((i + 1 for i, e in enumerate(flags) if e), None)
            horizon = first if first else len(flags)
            for n in range(lo, min(hi, horizon) + 1):
                risk += 1
                ev += (first == n)
        rows.append((lo, hi, risk, ev, ev / risk if risk else np.nan))
    return rows


print("=== H2: hazard including single-attempt cells (min_attempts=1) ===")
allcells = load_cells(min_attempts=1)
print(f"all cells: {len(allcells):,}  (vs {sum(1 for v in allcells.values() if len(v)>=2):,} with >=2)")
gpt = {k: v for k, v in allcells.items() if v[0][3] == "GPT_35_TURBO_0613"}
for label, cs in [("ALL cells", gpt),
                  (">=2 attempts only", {k: v for k, v in gpt.items() if len(v) >= 2})]:
    rows = hazard(cs)
    print(f"\n  {label}  (n={len(cs):,})")
    print(f"    {'attempt':>9s} {'at_risk':>9s} {'succ':>6s} {'hazard':>8s}")
    for lo, hi, r, e, h in rows:
        if r:
            lab = f"{lo}-{hi if hi < 10**9 else 'inf'}"
            print(f"    {lab:>9s} {r:9,d} {e:6,d} {100*h:7.3f}%")

print("\n=== H1: is attempt 1 a short probe? (GPT cells, >=4 attempts) ===")
L = defaultdict(list)
for k, v in gpt.items():
    if len(v) < 4:
        continue
    for i, (_, s, txt, _) in enumerate(v[:8]):
        L[i + 1].append(len(txt))
print(f"  {'attempt':>7s} {'n':>7s} {'median_chars':>13s} {'mean_chars':>11s}")
for i in sorted(L):
    a = np.array(L[i])
    print(f"  {i:7d} {len(a):7,d} {np.median(a):13.0f} {a.mean():11.0f}")

out = {"note": "see stdout"}
(RESULTS_DIR / "tt_anomaly_check.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

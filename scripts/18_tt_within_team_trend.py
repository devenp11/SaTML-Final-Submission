"""Same frailty-proof within-team test, on Tensor Trust (independent corpus).

Cell = (attacker, defense_id). Under "no learning" a cell's successes are a uniform
random size-s subset of its N attempts -> mean position exactly (N+1)/2, so the
centred statistic below has null expectation 0.
  < 0.5 successes early -> within-cell hazard declines ("failures don't teach")
  > 0.5 successes late  -> within-cell LEARNING
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tensortrust import load_cells
from config import RESULTS_DIR

rng = np.random.default_rng(0)
MIN_ATTEMPTS = 4
MIN_CELLS = 20

print("loading Tensor Trust...")
cells = load_cells(min_attempts=MIN_ATTEMPTS)
byllm = defaultdict(list)
for (atk, dfid), v in cells.items():
    byllm[v[0][3]].append(v)
print(f"cells with >={MIN_ATTEMPTS} attempts: {len(cells):,}")


def stat(v, drop_quitters):
    N = len(v)
    pos = [i + 1 for i, (_, s, _, _) in enumerate(v) if s]
    if not pos or N < MIN_ATTEMPTS:
        return None
    if drop_quitters and pos[-1] == N:
        return None
    # exactly centred: under the uniform null E[mean(pos)] = (N+1)/2
    return (np.mean(pos) - (N + 1) / 2) / N


def ci(a, n=3000):
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


report = {}
for drop in (False, True):
    label = "EXCLUDING quit-after-success" if drop else "ALL cells (conservative)"
    print(f"\n{'='*80}\nTENSOR TRUST within-cell success position — {label}\n{'='*80}")
    print(f"  {'llm':22s} {'cells':>7s} {'centred':>10s} {'95% CI':>20s} {'verdict':>8s}")
    pooled, rows = [], []
    for llm, vs in sorted(byllm.items()):
        a = np.array([x for x in (stat(v, drop) for v in vs) if x is not None])
        if len(a) < MIN_CELLS:
            continue
        lo, hi = ci(a)
        verdict = "EARLY" if hi < 0 else ("LATE" if lo > 0 else "ns")
        print(f"  {llm:22s} {len(a):7,d} {a.mean():+10.4f} [{lo:+.4f},{hi:+.4f}] {verdict:>8s}")
        rows.append({"llm": llm, "cells": len(a), "mean": float(a.mean()),
                     "ci": [float(lo), float(hi)], "verdict": verdict})
        pooled.append(a)
    if pooled:
        P = np.concatenate(pooled)
        lo, hi = ci(P)
        v = "EARLY (failures don't teach)" if hi < 0 else ("LATE (LEARNING)" if lo > 0 else "ns")
        print(f"\n  POOLED n={len(P):,} cells  mean={P.mean():+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]  -> {v}")
        report["quitters_dropped" if drop else "all"] = {
            "rows": rows, "pooled_n": int(len(P)), "pooled_mean": float(P.mean()),
            "pooled_ci": [float(lo), float(hi)], "verdict": v}

(RESULTS_DIR / "tt_within_team_trend.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
print(f"\nwrote {RESULTS_DIR/'tt_within_team_trend.json'}")

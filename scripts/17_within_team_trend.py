"""EXPERIMENT 1 — does the hazard rise WITHIN a team? (frailty-proof test of claim 1)

A declining POPULATION hazard is also what pure heterogeneity produces: high-hazard
teams break through early and leave the risk set. That is consistent with claim 1
("failures don't teach") but does not establish it. The threat is genuine within-team
learning MASKED by strong frailty, netting out to a population decline.

This test conditions on each team's own (N attempts, s successes), so team skill and
frailty cancel exactly. Under "no learning", the positions of a team's successes are a
uniform random size-s subset of 1..N, whose mean position is exactly (N+1)/2. The
statistic below subtracts that centre and divides by N, so its null expectation is 0.
  statistic < 0  -> successes early  -> within-team hazard DECLINES (claim 1 holds)
  statistic > 0  -> successes late   -> within-team LEARNING (claim 1 fails)

KNOWN CONSERVATIVE BIAS: teams that quit after succeeding put that success at position
N, pulling the statistic toward 1.0 -- i.e. AGAINST claim 1. Reported both ways.
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import iter_submissions
from config import RESULTS_DIR

rng = np.random.default_rng(0)
MIN_ATTEMPTS = 4
MIN_TEAMS = 15


def load(phase):
    d = defaultdict(lambda: defaultdict(list))
    for s in iter_submissions(phase=phase):
        d[(s["model"], s["defense"])][s["team_id"]].append((s["started_time"] or "", s["end_to_end"]))
    for cell in d.values():
        for t in cell:
            cell[t] = [e for _, e in sorted(cell[t], key=lambda x: x[0])]
    return d


print("loading both phases...")
DATA = {p: load(p) for p in (1, 2)}


def team_stats(flags, drop_quitters):
    N = len(flags)
    pos = [i + 1 for i, e in enumerate(flags) if e]
    if not pos or N < MIN_ATTEMPTS:
        return None
    if drop_quitters and pos[-1] == N:
        return None
    # exactly centred: under the uniform null E[mean(pos)] = (N+1)/2
    return (np.mean(pos) - (N + 1) / 2) / N


def ci(v, n=3000):
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


report = {}
for drop in (False, True):
    label = "EXCLUDING quit-after-success" if drop else "ALL teams (conservative)"
    print(f"\n{'='*86}\nWITHIN-TEAM SUCCESS POSITION — {label}\n{'='*86}")
    print(f"  {'ph':>2s} {'model':6s} {'defense':22s} {'teams':>6s} {'centred':>10s} {'95% CI':>20s} {'verdict':>8s}")
    rows, pooled = [], []
    for phase in (1, 2):
        for key in sorted(DATA[phase]):
            v = np.array([x for x in (team_stats(f, drop) for f in DATA[phase][key].values())
                          if x is not None])
            if len(v) < MIN_TEAMS:
                continue
            lo, hi = ci(v)
            verdict = "EARLY" if hi < 0 else ("LATE" if lo > 0 else "ns")
            print(f"  {phase:2d} {key[0]:6s} {key[1]:22s} {len(v):6d} {v.mean():+10.4f} "
                  f"[{lo:+.4f},{hi:+.4f}] {verdict:>8s}")
            rows.append({"phase": phase, "model": key[0], "defense": key[1], "teams": len(v),
                         "mean": float(v.mean()), "ci": [float(lo), float(hi)], "verdict": verdict})
            pooled.append(v)
    if pooled:
        P = np.concatenate(pooled)
        lo, hi = ci(P)
        v = "EARLY (claim 1 holds)" if hi < 0 else ("LATE (claim 1 FAILS)" if lo > 0 else "ns")
        n_early = sum(r["verdict"] == "EARLY" for r in rows)
        n_late = sum(r["verdict"] == "LATE" for r in rows)
        print(f"\n  POOLED n={len(P):,} teams  mean={P.mean():+.4f}  95% CI [{lo:+.4f},{hi:+.4f}]  -> {v}")
        print(f"  per-cell: {n_early} EARLY, {n_late} LATE, {len(rows)-n_early-n_late} ns  (of {len(rows)})")
        report["quitters_dropped" if drop else "all"] = {
            "cells": rows, "pooled_n": int(len(P)), "pooled_mean": float(P.mean()),
            "pooled_ci": [float(lo), float(hi)], "verdict": v,
            "n_early": n_early, "n_late": n_late}

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "within_team_trend.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
print(f"\nwrote {RESULTS_DIR/'within_team_trend.json'}")

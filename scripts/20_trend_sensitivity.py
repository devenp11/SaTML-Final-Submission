"""Sensitivity of the within-attacker trend to how the stopping rule is handled.

Attackers stop after cracking a target, so the LAST success sits at position N by
construction and inflates the mean position. Script 17/18 handled this by dropping any
cell whose last attempt was a success -- which discards ~92% of Tensor Trust cells and
invites the obvious reviewer objection that the survivors are a selected subset.

Estimator T (TRUNCATED) avoids the exclusion entirely: restrict each cell to attempts
1..L-1 where L is the position of the LAST success, and ask where the remaining s-1
successes fall in that window. Excluding L removes the terminal success that the
stopping rule places at the boundary by construction, which is the mechanical component
of the bias. It does not by itself make the window independent of the stopping process:
whether a cell is observed, and how long its window is, may still depend on earlier
events. Needs s>=2.

Reported side by side:
  A  all cells                      (biased LATE   -> conservative against learning)
  B  drop cells ending in success   (script 17/18; heavy exclusion)
  T  truncated window, s>=2         (no exclusion for stopping; the defensible one)
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import iter_submissions
from src.tensortrust import load_cells
from config import RESULTS_DIR

rng = np.random.default_rng(0)
MIN_ATT = 4


def est(flags, mode):
    """flags: chronological list of bools. -> exactly centred position statistic, or None.

    Under the null that the success positions are a uniform random subset of the window
    {1..M}, E[mean(pos)] = (M+1)/2. Subtracting that centre and dividing by the window
    length M gives a statistic with null expectation exactly 0, positive when successes
    fall late in the window and negative when they fall early.
    """
    N = len(flags)
    pos = [i + 1 for i, e in enumerate(flags) if e]
    if not pos or N < MIN_ATT:
        return None
    if mode == "A":
        return (np.mean(pos) - (N + 1) / 2) / N
    if mode == "B":
        if pos[-1] == N:
            return None
        return (np.mean(pos) - (N + 1) / 2) / N
    if mode == "T":
        if len(pos) < 2:
            return None
        L = pos[-1]                       # last success anchors the window
        W = L - 1                         # window is attempts 1..L-1
        inner = [p for p in pos[:-1]]     # successes strictly before L
        if W < MIN_ATT or not inner:
            return None
        return (np.mean(inner) - (W + 1) / 2) / W
    raise ValueError(mode)


def ci(a, n=3000):
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return np.percentile(bs, [2.5, 97.5])


def run(name, cellflags):
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    print(f"  {'estimator':38s} {'cells':>8s} {'centred':>10s} {'95% CI':>20s} {'verdict':>8s}")
    res = {}
    for mode, desc in [("A", "A  all cells (conservative, biased late)"),
                       ("B", "B  drop cells ending in success"),
                       ("T", "T  truncated window, s>=2  <-- primary")]:
        v = np.array([x for x in (est(f, mode) for f in cellflags) if x is not None])
        if len(v) < 20:
            print(f"  {desc:38s} {len(v):8,d}   too few")
            continue
        lo, hi = ci(v)
        verdict = "EARLY" if hi < 0 else ("LATE" if lo > 0 else "ns")
        print(f"  {desc:38s} {len(v):8,d} {v.mean():+10.4f} [{lo:+.4f},{hi:+.4f}] {verdict:>8s}")
        res[mode] = {"cells": int(len(v)), "mean": float(v.mean()),
                     "ci": [float(lo), float(hi)], "verdict": verdict}
    return res


out = {}

print("loading LLMail...")
lm = defaultdict(list)
for s in iter_submissions(phase=1):
    lm[(s["model"], s["defense"], s["team_id"])].append((s["started_time"] or "", s["end_to_end"]))
for s in iter_submissions(phase=2):
    lm[(s["model"], s["defense"], s["team_id"])].append((s["started_time"] or "", s["end_to_end"]))
lm_flags = [[e for _, e in sorted(v, key=lambda x: x[0])] for v in lm.values()]
out["llmail"] = run("LLMAIL-INJECT (both phases, cell = model x defense x team)", lm_flags)

print("\nloading Tensor Trust...")
tt = load_cells(min_attempts=MIN_ATT)
tt_flags = [[s for _, s, _, _ in v] for v in tt.values()]
out["tensortrust"] = run("TENSOR TRUST (cell = attacker x defense)", tt_flags)

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "trend_sensitivity.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(f"\nwrote {RESULTS_DIR/'trend_sensitivity.json'}")

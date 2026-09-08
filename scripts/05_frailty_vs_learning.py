"""Frailty or learning?  Two within-team tests that frailty CANNOT explain.

T1 INTER-SUCCESS INTERVALS: for teams with >=2 e2e successes on the same
   (model,defense), compare attempts-to-FIRST success vs gap to SECOND.
   Learning => second gap << first.  Pure frailty => gaps ~equal (memoryless).

T2 PER-LEVEL hazard slope: pooling levels confounds difficulty with attempt
   index. Recompute slope within each (level, model, defense).
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import iter_submissions
from config import RESULTS_DIR

def load(phase):
    d = defaultdict(lambda: defaultdict(list))
    for s in iter_submissions(phase=phase):
        d[(s["level"], s["model"], s["defense"])][s["team_id"]].append((s["started_time"] or "", s["end_to_end"]))
    for cell in d.values():
        for t in cell: cell[t] = [e for _, e in sorted(cell[t], key=lambda x: x[0])]
    return d

def slope_from_flags(cell):
    BINS=[(1,1),(2,3),(4,7),(8,15),(16,31),(32,63),(64,127),(128,255),(256,511),(512,10**9)]
    xs, ys = [], []
    for lo, hi in BINS:
        risk = ev = 0
        for flags in cell.values():
            first = next((i+1 for i,e in enumerate(flags) if e), None)
            horizon = first if first else len(flags)
            for n in range(lo, min(hi, horizon)+1):
                risk += 1; ev += (first == n)
        if risk >= 30 and ev > 0:
            xs.append(np.log(lo)); ys.append(np.log(ev/risk))
    return (float(np.polyfit(xs, ys, 1)[0]), len(xs)) if len(xs) >= 3 else (np.nan, len(xs))

out = {"T1": [], "T2": []}
for phase in (1, 2):
    data = load(phase)
    # ---- T1: pool across levels within (model, defense) at team level
    md = defaultdict(lambda: defaultdict(list))
    for (lvl, model, dfn), cell in data.items():
        for t, flags in cell.items(): md[(model, dfn)][t].extend(flags)   # chronological within level; ok for gap test
    print(f"\n=== PHASE {phase} — T1 inter-success intervals (teams with >=2 successes) ===")
    print(f"  {'model':6s} {'defense':22s} {'n_teams':>7s} {'med_first':>9s} {'med_2nd_gap':>11s} {'ratio':>6s} {'frac_2nd<first':>14s}")
    for (model, dfn), cell in sorted(md.items()):
        firsts, gaps = [], []
        for flags in cell.values():
            idx = [i+1 for i, e in enumerate(flags) if e]
            if len(idx) >= 2:
                firsts.append(idx[0]); gaps.append(idx[1] - idx[0])
        if len(firsts) >= 5:
            f, g = np.median(firsts), np.median(gaps)
            frac = float(np.mean(np.array(gaps) < np.array(firsts)))
            print(f"  {model:6s} {dfn:22s} {len(firsts):7d} {f:9.0f} {g:11.0f} {g/f:6.2f} {100*frac:13.0f}%")
            out["T1"].append({"phase": phase, "model": model, "defense": dfn, "n": len(firsts),
                              "med_first": f, "med_gap2": g, "frac_faster": frac})
    # ---- T2: per-level slopes
    print(f"\n=== PHASE {phase} — T2 hazard slope per (level, model, defense) ===")
    print(f"  {'lvl':>3s} {'model':6s} {'defense':22s} {'teams':>6s} {'slope':>7s} {'bins':>4s}")
    for (lvl, model, dfn), cell in sorted(data.items()):
        sl, nb = slope_from_flags(cell)
        if nb >= 3:
            print(f"  {lvl:3d} {model:6s} {dfn:22s} {len(cell):6d} {sl:+7.2f} {nb:4d}")
            out["T2"].append({"phase": phase, "level": lvl, "model": model, "defense": dfn, "slope": sl, "bins": nb})
sl = [r["slope"] for r in out["T2"]]
print(f"\nT2 summary: {len(sl)} cells; slopes negative in {sum(s<0 for s in sl)}/{len(sl)}; median slope {np.median(sl):+.2f}")
(RESULTS_DIR/"frailty_vs_learning.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

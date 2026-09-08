"""Do attackers LEARN against each defense? Hazard by attempt index.

For each (phase, model, defense): among teams still unbroken at attempt n,
what fraction break through AT attempt n?  A rising hazard = the defense
leaks information that attackers exploit. A flat hazard = each attempt is
effectively independent (defense is 'oracle-free').
Slope of log-hazard vs log-attempt is the learning-rate summary.
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import iter_submissions
from config import RESULTS_DIR

BINS = [(1,1),(2,3),(4,7),(8,15),(16,31),(32,63),(64,127),(128,255),(256,511),(512,10**9)]

def load(phase):
    d = defaultdict(lambda: defaultdict(list))
    for s in iter_submissions(phase=phase):
        d[(s["model"], s["defense"])][s["team_id"]].append((s["started_time"] or "", s["end_to_end"]))
    for cell in d.values():
        for t in cell: cell[t] = [e for _, e in sorted(cell[t], key=lambda x: x[0])]
    return d

def hazard_by_bin(cell):
    """returns list of (bin_label, at_risk_attempts, events, hazard)"""
    out = []
    for lo, hi in BINS:
        risk = ev = 0
        for flags in cell.values():
            first = next((i+1 for i,e in enumerate(flags) if e), None)
            horizon = first if first else len(flags)
            for n in range(lo, min(hi, horizon)+1):
                risk += 1
                if first == n: ev += 1
        out.append((f"{lo}-{hi if hi<10**9 else 'inf'}", risk, ev, ev/risk if risk else np.nan))
    return out

def slope(hz):
    xs, ys = [], []
    for (lab, risk, ev, h) in hz:
        if risk >= 30 and h > 0:
            lo = int(lab.split('-')[0]); xs.append(np.log(lo)); ys.append(np.log(h))
    if len(xs) < 3: return np.nan, len(xs)
    return float(np.polyfit(xs, ys, 1)[0]), len(xs)

report = {}
for phase in (1, 2):
    data = load(phase)
    for key in sorted(data):
        model, defense = key
        hz = hazard_by_bin(data[key]); sl, npts = slope(hz)
        tag = f"phase{phase}/{model}/{defense}"
        print(f"\n{tag}   teams={len(data[key])}   log-log slope={sl:+.3f} (n_bins={npts})")
        print(f"  {'attempt':>9s} {'at_risk':>8s} {'breaks':>7s} {'hazard':>8s}")
        for lab, risk, ev, h in hz:
            if risk: print(f"  {lab:>9s} {risk:8d} {ev:7d} {100*h:7.3f}%")
        report[tag] = {"slope": sl, "n_bins": npts,
                       "hazard": [(l, r, e, h) for l, r, e, h in hz]}
(RESULTS_DIR/"learning_curves.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
print("\nwrote", RESULTS_DIR/"learning_curves.json")

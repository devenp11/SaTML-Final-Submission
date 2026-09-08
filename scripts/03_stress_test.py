"""Decisive stress test of the metric-disagreement finding.

T1 REPLICATION   - does the ASR-vs-survival disagreement appear in BOTH phases?
T2 HORIZON       - or is it an artifact of picking S(10)?
T3 PAIRED        - within-team, effort-matched comparison. Kills confound 1
                   (self-selected allocation) by construction: only teams that
                   attacked BOTH defenses, truncated to equal attempt counts.
T4 MULTIPLICITY  - all pairwise tests, Holm-corrected.
"""
import sys, json, itertools
from pathlib import Path
from collections import defaultdict
from math import erfc, sqrt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import iter_submissions
from src.survival import km_at
from config import RESULTS_DIR

HORIZONS = [5, 10, 20, 50, 100]

def load(phase):
    """(model,defense) -> team -> chronological list of e2e flags"""
    d = defaultdict(lambda: defaultdict(list))
    for s in iter_submissions(phase=phase):
        d[(s["model"], s["defense"])][s["team_id"]].append(
            (s["started_time"] or "", s["end_to_end"]))
    for cell in d.values():
        for tid in cell:
            cell[tid] = [e for _, e in sorted(cell[tid], key=lambda x: x[0])]
    return d

def cell_asr(cell):
    tot = sum(len(v) for v in cell.values())
    return sum(sum(v) for v in cell.values()) / tot if tot else np.nan

def cell_surv(cell, h):
    times, events = [], []
    for flags in cell.values():
        first = next((i + 1 for i, e in enumerate(flags) if e), None)
        times.append(first if first else len(flags))
        events.append(1 if first else 0)
    return km_at(times, events, h)

def mcnemar(b, c):
    if b + c == 0: return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return float(chi2), float(erfc(sqrt(chi2 / 2.0)))

def holm(pvals):
    idx = np.argsort(pvals); m = len(pvals); adj = np.empty(m); run = 0.0
    for rank, i in enumerate(idx):
        run = max(run, (m - rank) * pvals[i]); adj[i] = min(1.0, run)
    return adj

def main():
    report = {}
    for phase in (1, 2):
        data = load(phase)
        for model in ("Phi-3", "GPT"):
            keys = sorted([k for k in data if k[0] == model], key=lambda k: k[1])
            if len(keys) < 3: continue
            tag = f"phase{phase}/{model}"
            print(f"\n{'='*94}\n{tag}\n{'='*94}")

            asr = {k[1]: cell_asr(data[k]) for k in keys}
            by_asr = sorted(asr, key=lambda d: asr[d])
            print(f"  ASR order       : {' > '.join(by_asr)}")

            # ---- T2 horizon sensitivity ----
            orders = {}
            for h in HORIZONS:
                sv = {k[1]: cell_surv(data[k], h) for k in keys}
                o = sorted(sv, key=lambda d: -sv[d]); orders[h] = o
                flag = "" if o == by_asr else "  <-- differs from ASR"
                print(f"  S({h:<3d}) order   : {' > '.join(o)}{flag}")
            stable = len({tuple(v) for v in orders.values()}) == 1
            print(f"  survival order stable across horizons? {'YES' if stable else 'NO'}")

            # ---- T3 within-team, effort-matched paired tests ----
            print(f"  --- paired (same teams, equal attempts) ---")
            pv, lab = [], []
            for (m1, d1), (m2, d2) in itertools.combinations(keys, 2):
                A, B = data[(m1, d1)], data[(m2, d2)]
                shared = set(A) & set(B)
                b = c = both = neither = 0
                for t in shared:
                    n = min(len(A[t]), len(B[t]))
                    if n == 0: continue
                    ba, bb = any(A[t][:n]), any(B[t][:n])
                    if ba and not bb: b += 1
                    elif bb and not ba: c += 1
                    elif ba and bb: both += 1
                    else: neither += 1
                if b + c == 0: continue
                chi2, p = mcnemar(b, c)
                pv.append(p); lab.append((d1, d2, len(shared), b, c, both, neither, chi2, p))
            if pv:
                adj = holm(np.array(pv))
                print(f"    {'A':22s} {'B':22s} {'teams':>6s} {'A-only':>7s} {'B-only':>7s} {'p':>9s} {'p_holm':>9s}")
                for (d1, d2, ns, b, c, bo, ne, chi2, p), pa in zip(lab, adj):
                    star = " *" if pa < 0.05 else ""
                    print(f"    {d1:22s} {d2:22s} {ns:6d} {b:7d} {c:7d} {p:9.4f} {pa:9.4f}{star}")
                sig = int((adj < 0.05).sum())
                print(f"    {sig}/{len(adj)} pairs significant after Holm correction")
            else:
                print("    no discordant pairs")
                sig = 0

            report[tag] = {"asr_order": by_asr,
                           "surv_orders": {str(h): orders[h] for h in HORIZONS},
                           "horizon_stable": stable,
                           "agrees_with_asr": {str(h): orders[h] == by_asr for h in HORIZONS},
                           "n_sig_paired": sig, "n_pairs": len(pv)}

    (RESULTS_DIR / "stress_test.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {RESULTS_DIR/'stress_test.json'}")

if __name__ == "__main__":
    main()

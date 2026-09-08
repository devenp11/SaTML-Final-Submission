"""Do ASR-based and effort-based evaluation rank defenses differently?

Unit of analysis = TEAM (the independent adversary). For each (model, defense):
  event time  = attempt index of a team's first end-to-end success
  censored    = team never succeeded; time = total attempts they made
Ranks defenses by ASR vs by survival, then bootstraps over teams to ask how
often the two orderings disagree.
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import iter_submissions
from src.survival import kaplan_meier, km_median, km_at, logrank, bootstrap_teams, ci
from config import RESULTS_DIR

PHASE, HORIZON, N_BOOT = 2, 10, 2000

def build():
    """-> {(model,defense): [ {attempts, time, event, n_succ}, ... per team ]}"""
    per_team = defaultdict(lambda: defaultdict(list))
    for s in iter_submissions(phase=PHASE):
        per_team[(s["model"], s["defense"])][s["team_id"]].append(
            (s["started_time"] or "", s["end_to_end"]))
    cells = {}
    for key, teams in per_team.items():
        recs = []
        for _tid, subs in teams.items():
            subs.sort(key=lambda x: x[0])          # chronological attempt order
            flags = [e for _, e in subs]
            n = len(flags)
            first = next((i + 1 for i, e in enumerate(flags) if e), None)
            recs.append({"attempts": n,
                         "time": first if first else n,
                         "event": 1 if first else 0,
                         "n_succ": sum(flags)})
        cells[key] = recs
    return cells

def asr(recs):
    tot = sum(r["attempts"] for r in recs)
    return sum(r["n_succ"] for r in recs) / tot if tot else np.nan

def surv(recs):
    return km_at([r["time"] for r in recs], [r["event"] for r in recs], HORIZON)

def main():
    cells = build()
    RESULTS_DIR.mkdir(exist_ok=True)
    report = {}

    for model in ["Phi-3", "GPT"]:
        keys = [k for k in cells if k[0] == model]
        print(f"\n{'='*96}\n{model}  (unit = team; survival at {HORIZON} attempts; {N_BOOT} bootstraps)\n{'='*96}")
        hdr = (f"{'defense':24s} {'teams':>6s} {'ASR%':>8s} {'ASR 95% CI':>18s} "
               f"{'S(10)':>7s} {'S(10) 95% CI':>18s} {'medAtt':>8s}")
        print(hdr); print("-"*len(hdr))
        rows = []
        for k in keys:
            recs = cells[k]
            a, s = asr(recs), surv(recs)
            a_ci = ci(bootstrap_teams(recs, asr, N_BOOT, seed=1))
            s_ci = ci(bootstrap_teams(recs, surv, N_BOOT, seed=2))
            med = km_median([r["time"] for r in recs], [r["event"] for r in recs])
            rows.append({"defense": k[1], "teams": len(recs), "asr": a, "asr_ci": a_ci,
                         "surv": s, "surv_ci": s_ci, "km_median": med})
            medtxt = "undef" if not np.isfinite(med) else f"{med:.0f}"
            print(f"{k[1]:24s} {len(recs):6d} {100*a:8.3f} "
                  f"[{100*a_ci[0]:6.3f},{100*a_ci[1]:6.3f}] {s:7.3f} "
                  f"[{s_ci[0]:6.3f},{s_ci[1]:6.3f}] {medtxt:>8s}")

        # rank agreement: ASR ascending (lower=better defense) vs survival descending
        by_asr  = [r["defense"] for r in sorted(rows, key=lambda r: r["asr"])]
        by_surv = [r["defense"] for r in sorted(rows, key=lambda r: -r["surv"])]
        print(f"\n  best->worst by ASR      : {' > '.join(by_asr)}")
        print(f"  best->worst by SURVIVAL : {' > '.join(by_surv)}")
        print(f"  same winner? {'YES' if by_asr[0]==by_surv[0] else 'NO  <-- REVERSAL'}")

        # bootstrap the whole ranking jointly
        rng = np.random.default_rng(7)
        disagree = top_diff = 0
        for _ in range(N_BOOT):
            bs = {}
            for k in keys:
                recs = cells[k]
                idx = rng.integers(0, len(recs), len(recs))
                r = [recs[i] for i in idx]
                bs[k[1]] = (asr(r), surv(r))
            ba = sorted(bs, key=lambda d: bs[d][0])
            bsv = sorted(bs, key=lambda d: -bs[d][1])
            if ba != bsv: disagree += 1
            if ba[0] != bsv[0]: top_diff += 1
        print(f"  bootstrap: orderings disagree in {100*disagree/N_BOOT:.1f}% of replicates; "
              f"top-ranked defense differs in {100*top_diff/N_BOOT:.1f}%")
        report[model] = {"rows": rows, "by_asr": by_asr, "by_surv": by_surv,
                         "pct_order_disagree": 100*disagree/N_BOOT,
                         "pct_top_differs": 100*top_diff/N_BOOT}

    # log-rank on the phase-2 pair flagged earlier
    k1, k2 = ("Phi-3","TaskTracker-Phi3"), ("Phi-3","PromptShield-v2")
    if k1 in cells and k2 in cells:
        c2, p = logrank([r["time"] for r in cells[k1]], [r["event"] for r in cells[k1]],
                        [r["time"] for r in cells[k2]], [r["event"] for r in cells[k2]])
        print(f"\nlog-rank  {k1[1]} vs {k2[1]} (Phi-3):  chi2={c2:.2f}  p={p:.4g}")
        report["logrank_TT3_vs_PS2_phi3"] = {"chi2": c2, "p": p}

    (RESULTS_DIR/"survival_ranks_phase2.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {RESULTS_DIR/'survival_ranks_phase2.json'}")

if __name__ == "__main__":
    main()

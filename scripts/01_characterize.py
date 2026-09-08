"""Per-defense outcomes on REAL adaptive human attacks (LLMail-Inject phase 2).

Produces the 'real attack' half of the rank comparison.
Metrics reported per defense (model held fixed):
  evasion_rate  - P(defense.undetected)            [detector-level]
  e2e_rate      - P(all objectives)                 [system-level]
  team_success  - fraction of teams with >=1 e2e    [skill-robust]
  median_effort - submissions before a team's first e2e (censored teams excluded)
"""
import sys, json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import iter_submissions
from config import RESULTS_DIR

PHASE = 2

def main():
    n = 0
    # (model, defense) -> counters
    agg = defaultdict(lambda: {"n": 0, "evaded": 0, "e2e": 0})
    # (model, defense, team) -> [attempts_seen, attempts_to_first_success or None]
    team = defaultdict(lambda: {"attempts": 0, "first": None, "e2e": 0})

    for s in iter_submissions(phase=PHASE):
        n += 1
        key = (s["model"], s["defense"])
        a = agg[key]
        a["n"] += 1
        a["evaded"] += s["evaded_defense"]
        a["e2e"] += s["end_to_end"]

        tk = (s["model"], s["defense"], s["team_id"])
        t = team[tk]
        t["attempts"] += 1
        if s["end_to_end"]:
            t["e2e"] += 1
            if t["first"] is None:
                t["first"] = t["attempts"]

    print(f"parsed submissions: {n:,}\n")

    # roll team stats up per (model, defense)
    teamroll = defaultdict(lambda: {"teams": 0, "succ": 0, "efforts": []})
    for (model, defense, _tid), t in team.items():
        tr = teamroll[(model, defense)]
        tr["teams"] += 1
        if t["first"] is not None:
            tr["succ"] += 1
            tr["efforts"].append(t["first"])

    def median(xs):
        xs = sorted(xs)
        if not xs:
            return float("nan")
        m = len(xs) // 2
        return float(xs[m]) if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2

    rows = []
    for key, a in agg.items():
        model, defense = key
        tr = teamroll[key]
        rows.append({
            "model": model,
            "defense": defense,
            "n": a["n"],
            "evasion_rate": a["evaded"] / a["n"] if a["n"] else float("nan"),
            "e2e_rate": a["e2e"] / a["n"] if a["n"] else float("nan"),
            "teams": tr["teams"],
            "team_success": tr["succ"] / tr["teams"] if tr["teams"] else float("nan"),
            "median_effort": median(tr["efforts"]),
        })

    rows.sort(key=lambda r: (r["model"], r["e2e_rate"]))
    hdr = f"{'model':6s} {'defense':24s} {'n':>8s} {'evade%':>8s} {'e2e%':>7s} {'teams':>6s} {'teamSucc%':>10s} {'medEffort':>10s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['model']:6s} {r['defense']:24s} {r['n']:8,d} "
              f"{100*r['evasion_rate']:8.2f} {100*r['e2e_rate']:7.3f} "
              f"{r['teams']:6d} {100*r['team_success']:10.2f} {r['median_effort']:10.1f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"real_attack_outcomes_phase{PHASE}.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main()

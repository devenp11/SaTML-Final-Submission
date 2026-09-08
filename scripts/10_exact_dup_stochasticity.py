"""Is the detector boundary STOCHASTIC?  P(pass | EXACT resubmission of a payload that passed).
Deterministic detector -> ~100%.  Stochastic (sampled LLM judge) -> well below.
Phase 1 full bodies. Flag = defense.undetected (detector-only) and e2e for comparison."""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import parse_scenario, OBJ_KEYS
from config import DATA_DIR, PHASE1_LEVELS, RESULTS_DIR
rng = np.random.default_rng(0)

cells = defaultdict(lambda: defaultdict(list))
with open(DATA_DIR / "phase1_succteams_full.jsonl", encoding="utf-8") as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        _, var = parse_scenario(r.get("scenario"))
        if var not in PHASE1_LEVELS: continue
        obj = r.get("objectives"); obj = json.loads(obj) if isinstance(obj, str) else (obj or {})
        und = bool(obj.get("defense.undetected", False)); e2e = all(bool(obj.get(k, False)) for k in OBJ_KEYS)
        cells[PHASE1_LEVELS[var]][r.get("team_id")].append(
            (r.get("started_time") or "", und, e2e, ((r.get("subject") or "") + "\n" + (r.get("body") or "")).strip()))

def run(flag_idx, label):
    print(f"\n=== P(pass | EXACT duplicate of a payload that passed)   [{label}] ===")
    print(f"  {'model':6s} {'defense':18s} {'teams':>5s} {'n_dups':>6s} {'pass%':>7s} {'95% CI':>15s}")
    res = {}
    for key in sorted(cells):
        per_team = []
        for team, subs in cells[key].items():
            subs.sort(key=lambda x: x[0])
            passed = {}          # payload text -> first index it passed
            n = s = 0
            for i, rec in enumerate(subs):
                txt, ok = rec[3], rec[flag_idx]
                if txt in passed:  # exact resubmission of something that previously passed
                    n += 1; s += ok
                elif ok:
                    passed[txt] = i
            if n: per_team.append((n, s))
        if len(per_team) < 5: continue
        N = sum(n for n, _ in per_team); S = sum(s for _, s in per_team)
        bs = []
        for _ in range(2000):
            idx = rng.integers(0, len(per_team), len(per_team))
            nn = sum(per_team[i][0] for i in idx); ss = sum(per_team[i][1] for i in idx)
            if nn: bs.append(ss / nn)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        print(f"  {key[0]:6s} {key[1]:18s} {len(per_team):5d} {N:6d} {100*S/N:6.1f}% [{100*lo:5.1f},{100*hi:5.1f}]")
        res[f"{key[0]}/{key[1]}"] = {"teams": len(per_team), "n_dups": N, "pass_rate": S/N, "ci": [float(lo), float(hi)]}
    return res

out = {"undetected": run(1, "flag = defense.undetected  (Spotlight always passes; ignore)"),
       "e2e":        run(2, "flag = end-to-end success  (includes LLM tool-call stochasticity)")}
(RESULTS_DIR/"exact_dup_stochasticity.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print("\nwrote", RESULTS_DIR/"exact_dup_stochasticity.json")

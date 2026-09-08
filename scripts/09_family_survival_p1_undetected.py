"""[DETECTOR-ONLY: flag = defense.undetected] Phase 1, full bodies, 101 successful teams. The mechanism test with power.

Per (model, defense):
  smooth[bin]     P(later submission succeeds | Jaccard to team's FIRST exploit in bin)
  family_survival P(success | Jaccard >= 0.7)   -- 'does the exploit family replay?'
  novelty_2nd     median Jaccard(1st success, 2nd success)
  n_teams, n_near (count of near-variant submissions)
Bootstrap CI on family_survival by resampling teams.
"""
import sys, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import parse_scenario, OBJ_KEYS
from config import DATA_DIR, PHASE1_LEVELS, RESULTS_DIR

def tokens(s): return set(re.findall(r"\w+", s.lower()))
BINS = [(0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0.3, 0.5), (0.0, 0.3)]
rng = np.random.default_rng(0)

cells = defaultdict(lambda: defaultdict(list))
with open(DATA_DIR / "phase1_succteams_full.jsonl", encoding="utf-8") as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        _, var = parse_scenario(r.get("scenario"))
        if var not in PHASE1_LEVELS: continue
        obj = r.get("objectives"); obj = json.loads(obj) if isinstance(obj, str) else (obj or {})
        e2e = bool(obj.get("defense.undetected", False))
        cells[PHASE1_LEVELS[var]][r.get("team_id")].append(
            (r.get("started_time") or "", e2e, tokens((r.get("subject") or "") + " " + (r.get("body") or ""))))

out = {}
hdr = f"{'model':6s} {'defense':18s} {'teams':>5s} {'n_near':>6s} {'famSurv%':>9s} {'95% CI':>15s} {'nov2nd':>7s} | {'[.9,1]':>7s} {'[.7,.9)':>7s} {'[.5,.7)':>7s} {'[.3,.5)':>7s} {'[0,.3)':>7s}"
print(hdr); print("-" * len(hdr))
for key in sorted(cells):
    per_team = []   # (near_n, near_s, bins[5][2], jac12)
    for team, subs in cells[key].items():
        subs.sort(key=lambda x: x[0])
        fi = next((i for i, (_, e, _) in enumerate(subs) if e), None)
        if fi is None: continue
        S = subs[fi][2]; bins = [[0, 0] for _ in BINS]; near = [0, 0]; jac12 = None
        for _, e, T in subs[fi+1:]:
            j = len(S & T) / len(S | T) if (S | T) else 1.0
            if e and jac12 is None: jac12 = j
            if j >= 0.7: near[0] += 1; near[1] += e
            for b, (lo, hi) in enumerate(BINS):
                if lo <= j < hi: bins[b][0] += 1; bins[b][1] += e; break
        per_team.append((near, bins, jac12))
    if len(per_team) < 5: continue
    N = sum(t[0][0] for t in per_team); Sx = sum(t[0][1] for t in per_team)
    fam = Sx / N if N else np.nan
    # team-level bootstrap on family survival
    bs = []
    for _ in range(2000):
        idx = rng.integers(0, len(per_team), len(per_team))
        n = sum(per_team[i][0][0] for i in idx); s = sum(per_team[i][0][1] for i in idx)
        if n: bs.append(s / n)
    lo, hi = (np.percentile(bs, 2.5), np.percentile(bs, 97.5)) if bs else (np.nan, np.nan)
    agg = [[sum(t[1][b][0] for t in per_team), sum(t[1][b][1] for t in per_team)] for b in range(len(BINS))]
    rates = [(s / n if n >= 20 else np.nan) for n, s in agg]
    novs = [t[2] for t in per_team if t[2] is not None]
    nov = np.median(novs) if novs else np.nan
    f = lambda x: f"{100*x:6.1f}%" if np.isfinite(x) else f"{'-':>7s}"
    print(f"{key[0]:6s} {key[1]:18s} {len(per_team):5d} {N:6d} {100*fam:8.2f}% [{100*lo:5.1f},{100*hi:5.1f}] {nov:7.2f} | " + " ".join(f(x) for x in rates))
    out[f"{key[0]}/{key[1]}"] = {"teams": len(per_team), "n_near": N, "family_survival": fam, "ci": [float(lo), float(hi)],
                                  "novelty_2nd_jaccard": float(nov) if np.isfinite(nov) else None,
                                  "bins": [(BINS[b][0], BINS[b][1], agg[b][0], agg[b][1]) for b in range(len(BINS))]}
(RESULTS_DIR/"family_survival_phase1_UNDETECTED.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print("\nwrote", RESULTS_DIR/"family_survival_phase1_UNDETECTED.json")

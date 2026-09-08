"""Three boring-explanation killers for the post-success speedup.

A REPLAY (phase 2, has bodies): is the 2nd successful payload a near-copy of the 1st?
  Jaccard on token sets. High similarity => 'learning' is just exploit reuse.
B WARMUP: teams that attacked >=2 defenses. Is first-gap on their SECOND defense
  short (interface warm-up) or long again (defense-specific)?
C NULL: per-team constant hazard at the team's own empirical rate. Simulate,
  condition on >=2 successes, get distribution of frac(2nd gap < 1st gap).
"""
import sys, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import parse_scenario, OBJ_KEYS
from config import DATA_DIR, PHASE2_LEVELS, PHASE1_LEVELS, RESULTS_DIR

def raw_iter(phase, want_body):
    levels = PHASE2_LEVELS if phase == 2 else PHASE1_LEVELS
    p = DATA_DIR / f"raw_submissions_phase{phase}.jsonl"
    if not p.exists(): p = DATA_DIR / f"phase{phase}_slim.jsonl"
    with open(p, encoding="utf-8") as f:
        for line in f:
            try: r = json.loads(line)
            except: continue
            lvl, var = parse_scenario(r.get("scenario"))
            if var not in levels: continue
            obj = r.get("objectives"); obj = json.loads(obj) if isinstance(obj, str) else (obj or {})
            e2e = all(bool(obj.get(k, False)) for k in OBJ_KEYS)
            yield (levels[var], r.get("team_id"), r.get("started_time") or "", e2e,
                   (r.get("subject","") + " " + r.get("body","")) if want_body else None)

def tokens(s): return set(re.findall(r"\w+", s.lower()))

out = {}
# ---------- A: replay on phase 2 ----------
cells = defaultdict(lambda: defaultdict(list))
for key, team, t, e2e, body in raw_iter(2, True):
    cells[key][team].append((t, e2e, body))
sims = []; exact = 0
for key, teams in cells.items():
    for team, subs in teams.items():
        subs.sort(key=lambda x: x[0])
        succ = [b for _, e, b in subs if e]
        if len(succ) >= 2:
            a, b = tokens(succ[0]), tokens(succ[1])
            j = len(a & b) / len(a | b) if (a | b) else 1.0
            sims.append(j); exact += (succ[0].strip() == succ[1].strip())
sims = np.array(sims)
print(f"A REPLAY (phase 2): {len(sims)} team-cells with >=2 successes")
print(f"   exact duplicate 2nd payload: {exact}/{len(sims)} ({100*exact/len(sims):.0f}%)")
print(f"   Jaccard(1st,2nd) median={np.median(sims):.2f}  >0.8: {100*np.mean(sims>0.8):.0f}%  >0.5: {100*np.mean(sims>0.5):.0f}%  <0.3: {100*np.mean(sims<0.3):.0f}%")
out["A"] = {"n": len(sims), "exact": exact, "jaccard_median": float(np.median(sims)),
            "frac_gt_0.8": float(np.mean(sims>0.8)), "frac_lt_0.3": float(np.mean(sims<0.3))}

# ---------- B + C on phase 1 (bigger) ----------
cells = defaultdict(lambda: defaultdict(list))
for key, team, t, e2e, _ in raw_iter(1, False):
    cells[key][team].append((t, e2e))
for key in cells:
    for team in cells[key]:
        cells[key][team] = [e for _, e in sorted(cells[key][team], key=lambda x: x[0])]

# B: per team, order the defenses they attacked by first-attempt time; first-gap on 1st vs 2nd defense
team_defs = defaultdict(list)   # team -> [(first_time, defense, flags)]
for key, team, t, e2e, _ in raw_iter(1, False):
    pass  # already loaded; rebuild with times below
tt = defaultdict(lambda: defaultdict(list))
for key, team, t, e2e, _ in raw_iter(1, False):
    tt[team][key].append((t, e2e))
first_on_1st, first_on_2nd = [], []
for team, bykey in tt.items():
    if len(bykey) < 2: continue
    order = sorted(bykey.items(), key=lambda kv: min(t for t, _ in kv[1]))
    gaps = []
    for key, subs in order[:2]:
        flags = [e for _, e in sorted(subs)]
        first = next((i+1 for i, e in enumerate(flags) if e), None)
        gaps.append(first)
    if gaps[0] and gaps[1]:
        first_on_1st.append(gaps[0]); first_on_2nd.append(gaps[1])
f1, f2 = np.array(first_on_1st), np.array(first_on_2nd)
print(f"\nB WARMUP (phase 1): {len(f1)} teams broke both their 1st and 2nd defense")
print(f"   median first-gap on 1st defense={np.median(f1):.0f}, on 2nd defense={np.median(f2):.0f}, frac(2nd<1st)={100*np.mean(f2<f1):.0f}%")
out["B"] = {"n": len(f1), "med_first_def": float(np.median(f1)), "med_second_def": float(np.median(f2)), "frac_faster": float(np.mean(f2<f1))}

# C: null simulation, per-team constant hazard
rng = np.random.default_rng(0)
obs_frac, null_fracs = {}, {}
for key, teams in cells.items():
    firsts, gaps = [], []
    for flags in teams.values():
        idx = [i+1 for i, e in enumerate(flags) if e]
        if len(idx) >= 2: firsts.append(idx[0]); gaps.append(idx[1]-idx[0])
    if len(firsts) < 5: continue
    obs = float(np.mean(np.array(gaps) < np.array(firsts)))
    sims = []
    for _ in range(2000):
        F, G = [], []
        for flags in teams.values():
            n, k = len(flags), sum(flags)
            if k == 0: continue
            sim = rng.random(n) < (k / n)
            idx = np.flatnonzero(sim) + 1
            if len(idx) >= 2: F.append(idx[0]); G.append(idx[1]-idx[0])
        if len(F) >= 5: sims.append(np.mean(np.array(G) < np.array(F)))
    sims = np.array(sims)
    pval = float(np.mean(sims >= obs)) if len(sims) else np.nan
    obs_frac[key] = obs; null_fracs[key] = (float(sims.mean()), pval)
print(f"\nC NULL (phase 1): frac(2nd gap < 1st gap), observed vs per-team-constant-hazard null")
print(f"   {'model':6s} {'defense':20s} {'observed':>9s} {'null_mean':>10s} {'p':>8s}")
for key in sorted(obs_frac):
    print(f"   {key[0]:6s} {key[1]:20s} {100*obs_frac[key]:8.0f}% {100*null_fracs[key][0]:9.0f}% {null_fracs[key][1]:8.4f}")
out["C"] = {f"{k[0]}/{k[1]}": {"obs": obs_frac[k], "null_mean": null_fracs[k][0], "p": null_fracs[k][1]} for k in obs_frac}
(RESULTS_DIR/"replay_warmup_null.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

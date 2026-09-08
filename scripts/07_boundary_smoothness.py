"""Local smoothness of each defense's decision boundary around FOUND exploits.

For each team's FIRST successful payload S in a (model,defense) cell, look at
every LATER submission by that team in the same cell. Bin by Jaccard(S, later)
and compute P(later succeeds). A 'smooth' defense: near-copies of a working
exploit keep working. A 'rough' one: they don't. Phase 2 (has bodies).
"""
import sys, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import parse_scenario, OBJ_KEYS
from config import DATA_DIR, PHASE2_LEVELS, RESULTS_DIR

def tokens(s): return set(re.findall(r"\w+", s.lower()))
BINS = [(0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0.3, 0.5), (0.0, 0.3)]

cells = defaultdict(lambda: defaultdict(list))
with open(DATA_DIR / "raw_submissions_phase2.jsonl", encoding="utf-8") as f:
    for line in f:
        try: r = json.loads(line)
        except: continue
        _, var = parse_scenario(r.get("scenario"))
        if var not in PHASE2_LEVELS: continue
        obj = r.get("objectives"); obj = json.loads(obj) if isinstance(obj, str) else (obj or {})
        e2e = all(bool(obj.get(k, False)) for k in OBJ_KEYS)
        cells[PHASE2_LEVELS[var]][r.get("team_id")].append(
            (r.get("started_time") or "", e2e, tokens((r.get("subject","") + " " + r.get("body","")))))

out = {}
print(f"{'model':6s} {'defense':22s} {'teams':>5s} | P(success | Jaccard to first exploit in bin)")
print(f"{'':6s} {'':22s} {'':5s} | {'[.9,1]':>9s} {'[.7,.9)':>9s} {'[.5,.7)':>9s} {'[.3,.5)':>9s} {'[0,.3)':>9s}   n(.9+)")
for key in sorted(cells):
    per_bin = [[0, 0] for _ in BINS]; nteams = 0
    for team, subs in cells[key].items():
        subs.sort(key=lambda x: x[0])
        fi = next((i for i, (_, e, _) in enumerate(subs) if e), None)
        if fi is None: continue
        nteams += 1; S = subs[fi][2]
        for _, e, T in subs[fi+1:]:
            j = len(S & T) / len(S | T) if (S | T) else 1.0
            for b, (lo, hi) in enumerate(BINS):
                if lo <= j < hi: per_bin[b][0] += 1; per_bin[b][1] += e; break
    if nteams < 3: continue
    rates = [(s / n if n >= 10 else np.nan) for n, s in per_bin]
    fmt = lambda x: f"{100*x:8.1f}%" if np.isfinite(x) else f"{'-':>9s}"
    print(f"{key[0]:6s} {key[1]:22s} {nteams:5d} | " + " ".join(fmt(x) for x in rates) + f"   {per_bin[0][0]:6d}")
    out[f"{key[0]}/{key[1]}"] = {"teams": nteams, "bins": [(lo, hi, n, s) for (lo, hi), (n, s) in zip(BINS, per_bin)]}
(RESULTS_DIR/"boundary_smoothness.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

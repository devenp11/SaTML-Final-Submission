"""FSR under the two admissible family-seed definitions (Section 6.2 sensitivity).

"P evaded defense D" can mean P got past the DETECTOR, or that P achieved END-TO-END
success (retrieved, evaded, and fired the tool with correct arguments). Both are
defensible and they give materially different numbers, so the paper reports the
sensitivity rather than choosing silently.

Writes results/fsr_seed_definition.json.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import parse_scenario, OBJ_KEYS
from config import DATA_DIR, PHASE1_LEVELS, RESULTS_DIR

rng = np.random.default_rng(0)
NEAR = 0.7
N_BOOT = 2000


def toks(s):
    return set(re.findall(r"\w+", (s or "").lower()))


def jac(a, b):
    return len(a & b) / len(a | b) if (a | b) else 1.0


cells = defaultdict(lambda: defaultdict(list))
with open(DATA_DIR / "phase1_succteams_full.jsonl", encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        _, var = parse_scenario(r.get("scenario"))
        if var not in PHASE1_LEVELS:
            continue
        o = r.get("objectives")
        o = json.loads(o) if isinstance(o, str) else (o or {})
        cells[PHASE1_LEVELS[var]][r.get("team_id")].append((
            r.get("started_time") or "",
            bool(o.get("defense.undetected", False)),                 # index 1
            all(bool(o.get(k, False)) for k in OBJ_KEYS),             # index 2
            ((r.get("subject") or "") + "\n" + (r.get("body") or "")).strip()))


def fsr(key, seed_idx):
    """seed_idx 1 = seed is a detector evasion; 2 = seed is an end-to-end success.
    Survival is always measured on the detector flag."""
    per_team = []
    for team, subs in cells[key].items():
        subs = sorted(subs, key=lambda x: x[0])
        fi = next((i for i, r in enumerate(subs) if r[seed_idx]), None)
        if fi is None:
            continue
        S = toks(subs[fi][3])
        n = s = 0
        for _, und, _, txt in subs[fi + 1:]:
            if jac(S, toks(txt)) >= NEAR:
                n += 1
                s += und
        if n:
            per_team.append((n, s))
    if not per_team:
        return float("nan"), (float("nan"),) * 2, 0, 0
    N = sum(a for a, _ in per_team)
    S_ = sum(b for _, b in per_team)
    bs = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(per_team), len(per_team))
        n = sum(per_team[i][0] for i in ix)
        s = sum(per_team[i][1] for i in ix)
        if n:
            bs.append(s / n)
    return S_ / N, tuple(np.percentile(bs, [2.5, 97.5])), N, len(per_team)


print(f"{'defense':26s} | {'SEED = evaded detector':>32s} | {'SEED = end-to-end success':>32s}")
res = {}
for key in sorted(cells):
    if "Spotlight" in key[1]:      # preventive: detector flag is TRUE by construction
        continue
    a = fsr(key, 1)
    b = fsr(key, 2)
    name = f"{key[0]}/{key[1]}"
    print(f"{name:26s} | {100*a[0]:6.1f}% [{100*a[1][0]:5.1f},{100*a[1][1]:5.1f}] n={a[2]:6,d} "
          f"| {100*b[0]:6.1f}% [{100*b[1][0]:5.1f},{100*b[1][1]:5.1f}] n={b[2]:6,d}")
    res[name] = {
        "seed_detector": {"fsr": a[0], "ci": list(a[1]), "n": a[2], "teams": a[3]},
        "seed_e2e": {"fsr": b[0], "ci": list(b[1]), "n": b[2], "teams": b[3]},
    }

for mode in ("seed_detector", "seed_e2e"):
    order = sorted(res, key=lambda k: res[k][mode]["fsr"])
    lowest = order[0]
    print(f"\n  ordering ({mode}): " + " < ".join(x.split("/")[1][:11] for x in order))
    print(f"  judge/all-combined lowest: {'YES' if ('Judge' in lowest or 'Combined' in lowest) else 'NO'}")

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "fsr_seed_definition.json").write_text(
    json.dumps(res, indent=1, default=float), encoding="utf-8")
print(f"\nwrote {RESULTS_DIR/'fsr_seed_definition.json'}")

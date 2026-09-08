"""R1 REDONE. Can community-level diffusion explain the within-attacker rise?

The first attempt at this (script 24, R1) was WRONG: inside a calendar bin it compared
high-attempt-index attempts to low-attempt-index attempts ACROSS attackers. Attackers
sitting at a high attempt index at any given moment are the ones who have been grinding
without success -- the frail ones -- so that test measures frailty, not diffusion, and
reintroduces the confound the paper exists to correct. It reported FALLS for that reason.

Correct design: keep the comparison WITHIN attacker, and remove the time for diffusion.
Restrict to attackers whose entire attempt sequence fits inside a short wall-clock
window (1h / 6h / 24h). If a technique needed to diffuse through the community, it
cannot do so inside a single sitting -- so a within-attacker rise that survives the
narrow windows is learning, not diffusion.
"""
import sys, json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import iter_submissions
from src.tensortrust import load_cells
from config import RESULTS_DIR

rng = np.random.default_rng(0)
MIN_ATT = 6
WINDOWS_H = [1, 6, 24, 10**6]      # last = no restriction


def parse(ts):
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None


def ci(a, n=3000):
    a = np.asarray(a, float)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return tuple(np.percentile(bs, [2.5, 97.5]))


def split_half(cells, max_hours):
    """cells: list of [(time, success), ...] chronological."""
    d = []
    for v in cells:
        ts = [parse(t) for t, _ in v]
        if any(x is None for x in ts):
            continue
        span = (ts[-1] - ts[0]).total_seconds() / 3600.0
        if span > max_hours:
            continue
        flags = [e for _, e in v]
        pos = [i + 1 for i, e in enumerate(flags) if e]
        if len(pos) < 2:
            continue
        W = pos[-1] - 1
        if W < MIN_ATT:
            continue
        w = np.array(flags[:W], dtype=int)
        if w.sum() == 0:
            continue
        h = len(w) // 2
        if len(w[h:2 * h]) == 0:
            continue
        d.append(w[h:2 * h].mean() - w[:h].mean())
    return np.array(d)


def run(name, cells):
    print(f"\n{name}")
    print(f"  {'attacker span':>16s} {'windows':>8s} {'2nd-1st':>10s} {'95% CI':>22s} {'verdict':>8s}")
    res = {}
    for H in WINDOWS_H:
        d = split_half(cells, H)
        lab = "unrestricted" if H > 10**5 else f"<= {H}h"
        if len(d) < 25:
            print(f"  {lab:>16s} {len(d):8,d}   too few")
            continue
        lo, hi = ci(d)
        v = "RISES" if lo > 0 else ("FALLS" if hi < 0 else "ns")
        print(f"  {lab:>16s} {len(d):8,d} {d.mean():+10.5f} [{lo:+.5f},{hi:+.5f}] {v:>8s}")
        res[lab] = {"windows": int(len(d)), "mean": float(d.mean()),
                    "ci": [float(lo), float(hi)], "verdict": v}
    return res


out = {}
print("loading LLMail...")
lm = defaultdict(list)
for ph in (1, 2):
    for s in iter_submissions(phase=ph):
        lm[(s["model"], s["defense"], s["team_id"])].append((s["started_time"] or "", s["end_to_end"]))
out["llmail"] = run("LLMAIL — within-attacker split-half, restricted by attempt span",
                    [sorted(v, key=lambda x: x[0]) for v in lm.values()])

print("\nloading Tensor Trust...")
tt = [[(t, s) for t, s, _, _ in v] for v in load_cells(min_attempts=MIN_ATT).values()]
out["tensortrust"] = run("TENSOR TRUST — same", tt)

print("\nInterpretation: a rise that persists at <=1h/<=6h cannot be community diffusion,")
print("because there is no time for a technique to spread between those attempts.")

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "calendar_confound.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(f"\nwrote {RESULTS_DIR/'calendar_confound.json'}")

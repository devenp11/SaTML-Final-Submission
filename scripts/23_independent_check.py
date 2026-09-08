"""INDEPENDENT re-derivation of the headline by a different estimator.

Scripts 17/20 use the mean normalised success POSITION. Script 21 fits a power-law
hazard by conditional likelihood. Both are position-based and share assumptions, so
agreement between them is weak evidence.

This uses a SPLIT-HALF paired rate comparison instead: inside each attacker's window,
compare the empirical success RATE in the first half of attempts to the second half.
Same attacker on both sides, so frailty cancels; no parametric form; no reliance on the
position statistic.

    rate_2nd > rate_1st  ->  within-attacker hazard RISES (learning)

Also runs a WITHIN-CELL LABEL PERMUTATION null: shuffling each attacker's own success
labels across their own attempts destroys any within-attacker time structure while
preserving N, s, and every between-attacker difference. If the observed split-half gap
sits outside that null, the effect cannot be frailty.
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import iter_submissions
from src.tensortrust import load_cells
from config import RESULTS_DIR

rng = np.random.default_rng(0)
MIN_ATT = 6
N_PERM = 2000


def windows(cellflags):
    """Truncate at the last success so the stopping rule cannot bias the window."""
    out = []
    for flags in cellflags:
        pos = [i + 1 for i, e in enumerate(flags) if e]
        if len(pos) < 2:
            continue
        W = pos[-1] - 1
        if W < MIN_ATT:
            continue
        w = flags[:W]
        if sum(w) == 0:
            continue
        out.append(np.array(w, dtype=int))
    return out


def split_half(w):
    h = len(w) // 2
    return w[:h].mean(), w[h:2 * h].mean() if len(w[h:2 * h]) else np.nan


def analyse(name, cellflags):
    W = windows(cellflags)
    if len(W) < 30:
        print(f"{name}: too few windows ({len(W)})")
        return None
    d = np.array([b - a for a, b in (split_half(w) for w in W) if np.isfinite(b)])
    obs = d.mean()
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(3000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])

    # within-cell label permutation null: destroys time structure, keeps N and s
    null = []
    for _ in range(N_PERM):
        vals = []
        for w in W:
            p = rng.permutation(w)
            a, b = split_half(p)
            if np.isfinite(b):
                vals.append(b - a)
        null.append(np.mean(vals))
    null = np.array(null)
    pval = (np.sum(null >= obs) + 1) / (N_PERM + 1)

    print(f"\n{name}")
    print(f"  windows                       : {len(d):,}")
    print(f"  mean(rate 2nd half - 1st half): {obs:+.5f}   95% CI [{lo:+.5f}, {hi:+.5f}]")
    print(f"  permutation null mean         : {null.mean():+.5f}  (sd {null.std():.5f})")
    print(f"  one-sided p vs permutation    : {pval:.4f}")
    verdict = ("RISES (learning)" if lo > 0 and pval < 0.05
               else "FALLS" if hi < 0 else "not significant")
    print(f"  -> within-attacker hazard {verdict}")
    return {"windows": int(len(d)), "obs": float(obs), "ci": [float(lo), float(hi)],
            "null_mean": float(null.mean()), "p": float(pval), "verdict": verdict}


out = {}
print("loading LLMail (both phases)...")
lm = defaultdict(list)
for ph in (1, 2):
    for s in iter_submissions(phase=ph):
        lm[(s["model"], s["defense"], s["team_id"])].append((s["started_time"] or "", s["end_to_end"]))
out["llmail"] = analyse("LLMAIL-INJECT  (split-half, permutation null)",
                        [[e for _, e in sorted(v, key=lambda x: x[0])] for v in lm.values()])

print("\nloading Tensor Trust...")
out["tensortrust"] = analyse("TENSOR TRUST  (split-half, permutation null)",
                             [[s for _, s, _, _ in v] for v in load_cells(min_attempts=MIN_ATT).values()])

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "independent_check.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(f"\nwrote {RESULTS_DIR/'independent_check.json'}")

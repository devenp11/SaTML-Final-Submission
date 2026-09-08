"""Translate the within-attacker trend into an interpretable hazard ratio.

"mean normalised success position = 0.625" is not a quantity a reader can feel.
This fits the within-cell hazard shape

    h_c(n) = h_c * n^beta

by CONDITIONAL likelihood: conditioning on each cell's own success count, the cell
intercept h_c cancels exactly, so beta is estimated free of attacker skill/frailty.
(Breslow handling of multiple successes within a cell.)

  loglik(beta) = sum_c [ beta*log k_c  -  log sum_{n=1..W_c} n^beta ]

RESTRICTED TO CELLS WITH EXACTLY ONE SUCCESS IN THE WINDOW, where this conditional
likelihood is EXACT. With several successes per window the Breslow independence
approximation degrades badly: LLMail has windows holding up to 554 successes, and
those cells contribute one likelihood term each, swamping the clean cells and biasing
beta toward 0 (a dense window looks uniform). Validated by synthetic recovery.

beta > 0  -> hazard RISES with attempt index within an attacker (learning)
Reported as the hazard ratio per DOUBLING of attempt index, 2^beta.

Window W_c = (position of last success) - 1, matching estimator T in script 20, so the
attacker's stopping rule cannot influence the window.
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import iter_submissions
from src.tensortrust import load_cells
from config import RESULTS_DIR

rng = np.random.default_rng(0)
MIN_ATT = 4


def cells_to_windows(cellflags):
    """-> list of (W, [success positions strictly inside 1..W])"""
    out = []
    for flags in cellflags:
        N = len(flags)
        pos = [i + 1 for i, e in enumerate(flags) if e]
        if len(pos) < 2 or N < MIN_ATT:
            continue
        W = pos[-1] - 1
        inner = [p for p in pos[:-1] if p <= W]
        # exactly one success inside the window -> exact conditional likelihood
        if W >= MIN_ATT and len(inner) == 1:
            out.append((W, inner))
    return out


def negll(beta, data, logW_cache):
    ll = 0.0
    for W, S in data:
        n = np.arange(1, W + 1, dtype=float)
        denom = np.log(np.power(n, beta).sum())
        ll += sum(beta * np.log(k) - denom for k in S)
    return -ll


def fit(data):
    r = minimize_scalar(lambda b: negll(b, data, None), bounds=(-3.0, 3.0), method="bounded")
    return float(r.x)


def run(name, cellflags, n_boot=200):
    data = cells_to_windows(cellflags)
    if len(data) < 30:
        print(f"{name}: too few cells ({len(data)})")
        return None
    beta = fit(data)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(data), len(data))
        bs.append(fit([data[i] for i in idx]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    hr, hlo, hhi = 2 ** beta, 2 ** lo, 2 ** hi
    print(f"\n{name}")
    print(f"  cells fitted            : {len(data):,}")
    print(f"  beta (log-log slope)    : {beta:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  hazard ratio / doubling : {hr:.3f}x   95% CI [{hlo:.3f}, {hhi:.3f}]")
    print(f"  -> within one attacker, going from attempt n to attempt 2n multiplies the")
    print(f"     per-attempt success probability by {hr:.2f}x  ({100*(hr-1):+.0f}%)")
    return {"cells": len(data), "beta": beta, "beta_ci": [float(lo), float(hi)],
            "hr_per_doubling": hr, "hr_ci": [float(hlo), float(hhi)]}


out = {}
print("loading LLMail (both phases)...")
lm = defaultdict(list)
for ph in (1, 2):
    for s in iter_submissions(phase=ph):
        lm[(s["model"], s["defense"], s["team_id"])].append((s["started_time"] or "", s["end_to_end"]))
out["llmail"] = run("LLMAIL-INJECT", [[e for _, e in sorted(v, key=lambda x: x[0])] for v in lm.values()])

print("\nloading Tensor Trust...")
tt = load_cells(min_attempts=MIN_ATT)
out["tensortrust"] = run("TENSOR TRUST", [[s for _, s, _, _ in v] for v in tt.values()])

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "hazard_ratio.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(f"\nwrote {RESULTS_DIR/'hazard_ratio.json'}")

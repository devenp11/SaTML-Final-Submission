"""Reviewer response (weaknesses 1, 2): exact conditional logistic regression with
attacker fixed effects, on ALL windows with >= 1 inner success.

Model, per cell c with window attempts n = 1..W_c and success set S_c (|S_c| = s_c):

    logit P(success at n) = alpha_c + beta * log(n)

The fitted quantity 2**beta is therefore an ODDS RATIO per doubling of the
attempt index, not a hazard ratio and not exactly a probability ratio.

Conditioning on s_c eliminates alpha_c EXACTLY (no Breslow approximation):

    L_c(beta) = exp(beta * sum_{k in S_c} log k)  /  sum_{|A| = s_c} exp(beta * sum_{k in A} log k)

The denominator sums over every size-s_c subset of the window. Computed by the
standard clogit dynamic programme in log space, O(W_c * s_c) per cell, so it is exact
for every cell including LLMail windows holding hundreds of successes (which broke the
Breslow approximation in script 21).

Why zero-success cells cannot be included in ANY fixed-effects design: with s_c = 0 the
conditional likelihood is identically 1 for all beta; the cell carries no information
about the within-cell trend. Same for s_c = W_c. This is a property of the estimand,
not of our implementation. Reported explicitly.

CI: profile likelihood, 2*(l(beta_hat) - l(beta)) = 3.841. No bootstrap needed.
Reported on (T) truncated windows [primary; stopping rule cannot touch them] and on
(A) full sequences [biased late by stopping; shown for completeness].
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.optimize import minimize_scalar, brentq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import iter_submissions
from src.tensortrust import load_cells
from config import RESULTS_DIR

MIN_ATT = 4


def windows(cellflags, mode):
    """-> list of (W, sorted success positions inside 1..W). Requires 1 <= s < W."""
    out = []
    for flags in cellflags:
        N = len(flags)
        pos = [i + 1 for i, e in enumerate(flags) if e]
        if not pos or N < MIN_ATT:
            continue
        if mode == "T":
            if len(pos) < 2:
                continue
            W = pos[-1] - 1
            S = [p for p in pos[:-1] if p <= W]
        else:
            W, S = N, pos
        if W >= MIN_ATT and 0 < len(S) < W:
            out.append((W, S))
    return out


def cell_loglik(beta, W, S):
    """Exact conditional log-likelihood for one cell."""
    x = beta * np.log(np.arange(1, W + 1, dtype=float))       # beta * log n
    s = len(S)
    # log-space DP over subsets: f[m] = log sum over size-m subsets of processed attempts
    f = np.full(s + 1, -np.inf)
    f[0] = 0.0
    for j in range(W):
        # f_new[m] = logaddexp(f[m], f[m-1] + x_j)  for m >= 1, vectorised
        f[1:] = np.logaddexp(f[1:], f[:-1] + x[j])
    num = x[np.array(S) - 1].sum()
    return num - f[s]


def total_loglik(beta, data):
    return sum(cell_loglik(beta, W, S) for W, S in data)


def fit(data):
    r = minimize_scalar(lambda b: -total_loglik(b, data), bounds=(-3.0, 3.0), method="bounded",
                        options={"xatol": 1e-4})
    bh, lmax = float(r.x), -float(r.fun)
    crit = lmax - 3.841 / 2.0
    g = lambda b: total_loglik(b, data) - crit
    lo = brentq(g, -3.0, bh) if g(-3.0) < 0 else -3.0
    hi = brentq(g, bh, 3.0) if g(3.0) < 0 else 3.0
    return bh, lo, hi, lmax


def run(name, cellflags):
    print(f"\n{'='*84}\n{name}\n{'='*84}")
    res = {}
    n_all = len(cellflags)
    n_zero = sum(1 for f in cellflags if not any(f))
    print(f"  cells total {n_all:,}; with zero successes {n_zero:,} ({100*n_zero/n_all:.1f}%) "
          f"-> uninformative for any within-cell trend (likelihood constant in beta)")
    for mode, label in (("T", "T  truncated window (primary)"), ("A", "A  full sequence (stopping-biased)")):
        data = windows(cellflags, mode)
        ns = np.array([len(S) for _, S in data])
        bh, lo, hi, lmax = fit(data)
        hr = 2 ** bh
        sig = "RISES" if lo > 0 else ("FALLS" if hi < 0 else "ns")
        print(f"  {label}")
        print(f"     windows {len(data):,}  (successes/window: median {np.median(ns):.0f}, max {ns.max()})")
        print(f"     beta = {bh:+.4f}   95% profile CI [{lo:+.4f}, {hi:+.4f}]   -> {sig}")
        print(f"     odds ratio per doubling = {hr:.3f}x   [{2**lo:.3f}, {2**hi:.3f}]")
        res[mode] = {"windows": int(len(data)), "beta": bh, "ci": [lo, hi],
                     "or_per_doubling": hr, "or_ci": [2 ** lo, 2 ** hi], "verdict": sig,
                     "max_successes_in_window": int(ns.max())}
    res["cells_total"] = n_all
    res["cells_zero_success"] = n_zero
    return res


out = {}
print("loading LLMail (both phases)...")
lm = defaultdict(list)
for ph in (1, 2):
    for s in iter_submissions(phase=ph):
        lm[(s["model"], s["defense"], s["team_id"])].append((s["started_time"] or "", s["end_to_end"]))
out["llmail"] = run("LLMAIL-INJECT — exact clogit, attacker fixed effects",
                    [[e for _, e in sorted(v, key=lambda x: x[0])] for v in lm.values()])

print("\nloading Tensor Trust...")
out["tensortrust"] = run("TENSOR TRUST — exact clogit, attacker fixed effects",
                         [[s for _, s, _, _ in v] for v in load_cells(min_attempts=1).values()])

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "clogit_fixed_effects.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(f"\nwrote {RESULTS_DIR/'clogit_fixed_effects.json'}")

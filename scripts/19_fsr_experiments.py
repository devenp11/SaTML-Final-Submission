"""EXPERIMENTS 2-4 on claim 3 (family survival varies across defenses).

E2 REPLICATION : per-defense FSR distribution in Tensor Trust (thousands of defenses).
                 Does 15x spread replicate, or is it an artifact of 4 hand-built defenses?
E3 PAIRED      : within-team FSR comparison across LLMail defenses. Kills the
                 self-selection confound (teams chose which defense to attack).
E4 EXPLANATION : does FSR track the edit distance attackers need? If high-FSR defenses
                 are simply ones where small mutations suffice, FSR is a property of the
                 attack surface rather than of the decision boundary.
"""
import sys, json, re, itertools
from pathlib import Path
from collections import defaultdict
from math import erfc, sqrt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import parse_scenario, OBJ_KEYS
from src.tensortrust import load_cells, tokens as tt_tokens
from config import DATA_DIR, PHASE1_LEVELS, RESULTS_DIR

rng = np.random.default_rng(0)
NEAR = 0.7
out = {}


def toks(s):
    return set(re.findall(r"\w+", (s or "").lower()))


def ci(vals, n=2000):
    if not len(vals):
        return (np.nan, np.nan)
    bs = [np.mean(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(n)]
    return tuple(np.percentile(bs, [2.5, 97.5]))


# ─────────────────────────── E2: Tensor Trust per-defense FSR ───────────────────────────
print("=" * 84)
print("E2  Tensor Trust: per-defense FSR distribution")
print("=" * 84)
cells = load_cells()
per_def = defaultdict(lambda: [0, 0])
for (atk, dfid), v in cells.items():
    fi = next((i for i, (_, s, _, _) in enumerate(v) if s), None)
    if fi is None:
        continue
    S = tt_tokens(v[fi][2])
    for _, s, txt, _ in v[fi + 1:]:
        T = tt_tokens(txt)
        j = len(S & T) / len(S | T) if (S | T) else 1.0
        if j >= NEAR:
            per_def[dfid][0] += 1
            per_def[dfid][1] += s
fsrs = np.array([s / n for n, s in per_def.values() if n >= 10])
print(f"  defenses with >=10 near-variants: {len(fsrs):,}")
if len(fsrs):
    qs = np.percentile(fsrs, [5, 25, 50, 75, 95])
    print(f"  FSR distribution: p5={100*qs[0]:.1f}%  p25={100*qs[1]:.1f}%  median={100*qs[2]:.1f}%  "
          f"p75={100*qs[3]:.1f}%  p95={100*qs[4]:.1f}%")
    print(f"  spread p95/p5 = {qs[4]/max(qs[0],1e-9):.1f}x   (LLMail across 4 defenses: ~15x)")
    print(f"  fraction at exactly 0: {100*np.mean(fsrs==0):.1f}%   at exactly 1: {100*np.mean(fsrs==1):.1f}%")
    out["E2"] = {"n_defenses": int(len(fsrs)), "p5": float(qs[0]), "p25": float(qs[1]),
                 "median": float(qs[2]), "p75": float(qs[3]), "p95": float(qs[4]),
                 "frac_zero": float(np.mean(fsrs == 0)), "frac_one": float(np.mean(fsrs == 1))}

# ──────────────────────── load LLMail phase 1 with bodies (E3, E4) ────────────────────────
print("\nloading LLMail phase 1 (full bodies)...")
lc = defaultdict(lambda: defaultdict(list))
with open(DATA_DIR / "phase1_succteams_full.jsonl", encoding="utf-8") as f:
    for line in f:
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        _, var = parse_scenario(r.get("scenario"))
        if var not in PHASE1_LEVELS:
            continue
        obj = r.get("objectives")
        obj = json.loads(obj) if isinstance(obj, str) else (obj or {})
        lc[PHASE1_LEVELS[var]][r.get("team_id")].append((
            r.get("started_time") or "",
            all(bool(obj.get(k, False)) for k in OBJ_KEYS),
            ((r.get("subject") or "") + "\n" + (r.get("body") or "")).strip()))


def team_fsr(subs):
    """-> (n_near, n_survived, median_jaccard_of_survivors) for one team-cell."""
    subs.sort(key=lambda x: x[0])
    fi = next((i for i, (_, e, _) in enumerate(subs) if e), None)
    if fi is None:
        return None
    S = toks(subs[fi][2])
    n = s = 0
    surv_j = []
    for _, e, txt in subs[fi + 1:]:
        T = toks(txt)
        j = len(S & T) / len(S | T) if (S | T) else 1.0
        if j >= NEAR:
            n += 1
            s += e
            if e:
                surv_j.append(j)
    if n == 0:
        return None
    return (n, s, float(np.median(surv_j)) if surv_j else np.nan)


cellfsr = {}
for key, teams in lc.items():
    d = {}
    for t, subs in teams.items():
        r = team_fsr(list(subs))
        if r:
            d[t] = r
    cellfsr[key] = d

# ───────────────────────── E3: within-team paired FSR across defenses ─────────────────────
print("\n" + "=" * 84)
print("E3  Within-team paired FSR (same teams, both defenses)")
print("=" * 84)
print(f"  {'A':22s} {'B':22s} {'teams':>5s} {'FSR_A':>7s} {'FSR_B':>7s} {'diff':>8s} {'95% CI':>18s}")
pairs = []
for (m1, d1), (m2, d2) in itertools.combinations(sorted(cellfsr), 2):
    if m1 != m2:
        continue
    A, B = cellfsr[(m1, d1)], cellfsr[(m2, d2)]
    shared = sorted(set(A) & set(B))
    if len(shared) < 8:
        continue
    da = np.array([A[t][1] / A[t][0] for t in shared])
    db = np.array([B[t][1] / B[t][0] for t in shared])
    diff = da - db
    lo, hi = ci(diff)
    sig = "*" if (lo > 0 or hi < 0) else ""
    print(f"  {m1[:3]+'/'+d1:22s} {m2[:3]+'/'+d2:22s} {len(shared):5d} "
          f"{100*da.mean():6.1f}% {100*db.mean():6.1f}% {100*diff.mean():+7.1f}% "
          f"[{100*lo:+.1f},{100*hi:+.1f}]{sig}")
    pairs.append({"model": m1, "A": d1, "B": d2, "teams": len(shared),
                  "fsr_A": float(da.mean()), "fsr_B": float(db.mean()),
                  "diff": float(diff.mean()), "ci": [float(lo), float(hi)],
                  "sig": bool(lo > 0 or hi < 0)})
nsig = sum(p["sig"] for p in pairs)
print(f"\n  {nsig}/{len(pairs)} defense pairs differ significantly in PAIRED FSR")
out["E3"] = {"pairs": pairs, "n_sig": nsig, "n_pairs": len(pairs)}

# ───────────────────────── E4: does FSR track required edit distance? ─────────────────────
print("\n" + "=" * 84)
print("E4  Is FSR explained by how similar surviving variants must be?")
print("=" * 84)
print(f"  {'model':6s} {'defense':22s} {'teams':>5s} {'FSR':>7s} {'median J of survivors':>22s}")
xs, ys = [], []
for key in sorted(cellfsr):
    d = cellfsr[key]
    if len(d) < 5:
        continue
    N = sum(v[0] for v in d.values())
    S = sum(v[1] for v in d.values())
    js = [v[2] for v in d.values() if np.isfinite(v[2])]
    if not js or N == 0:
        continue
    fsr, mj = S / N, float(np.median(js))
    print(f"  {key[0]:6s} {key[1]:22s} {len(d):5d} {100*fsr:6.2f}% {mj:22.3f}")
    xs.append(mj)
    ys.append(fsr)
if len(xs) >= 4:
    r = float(np.corrcoef(xs, ys)[0, 1])
    print(f"\n  correlation(median survivor Jaccard, FSR) across {len(xs)} defense-cells: r = {r:+.3f}")
    print("  NOTE: n is tiny. This is descriptive, not a test.")
    out["E4"] = {"n": len(xs), "pearson_r": r,
                 "points": [{"jaccard": a, "fsr": b} for a, b in zip(xs, ys)]}

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "fsr_experiments.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(f"\nwrote {RESULTS_DIR/'fsr_experiments.json'}")

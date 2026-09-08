"""REPLICATION on Tensor Trust v2 (415k attacks, 3.2k attackers, 4.4k defenses, 3 LLMs).

C1 hazard by attempt index    -> do failures teach?
C2 inter-success gaps         -> do successes teach?
C3 family survival per LLM    -> P(near-variant succeeds | one did)
C4 exact-duplicate pass rate  -> boundary stochasticity
C5 MECHANISM: across individual defenses, does determinism predict family survival?
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.tensortrust import load_cells, tokens
from config import RESULTS_DIR

rng = np.random.default_rng(0)
BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 10**9)]

print("loading...")
cells = load_cells()
print(f"cells (attacker x defense, >=2 attempts): {len(cells):,}")

byllm = defaultdict(dict)
for k, v in cells.items():
    byllm[v[0][3]][k] = v


def hazard(cs):
    rows = []
    for lo, hi in BINS:
        risk = ev = 0
        for v in cs.values():
            flags = [s for _, s, _, _ in v]
            first = next((i + 1 for i, e in enumerate(flags) if e), None)
            horizon = first if first else len(flags)
            for n in range(lo, min(hi, horizon) + 1):
                risk += 1
                ev += (first == n)
        rows.append((lo, hi, risk, ev, ev / risk if risk else np.nan))
    xs = [np.log(lo) for lo, _, r, e, h in rows if r >= 50 and h > 0]
    ys = [np.log(h) for lo, _, r, e, h in rows if r >= 50 and h > 0]
    sl = float(np.polyfit(xs, ys, 1)[0]) if len(xs) >= 3 else np.nan
    return rows, sl


print("\n=== C1  hazard by attempt index ===")
res = {}
for llm, cs in sorted(byllm.items()):
    rows, sl = hazard(cs)
    print(f"\n{llm}   cells={len(cs):,}   log-log slope={sl:+.3f}")
    print(f"  {'attempt':>9s} {'at_risk':>9s} {'succ':>6s} {'hazard':>8s}")
    for lo, hi, r, e, h in rows:
        if r:
            lab = f"{lo}-{hi if hi < 10**9 else 'inf'}"
            print(f"  {lab:>9s} {r:9,d} {e:6,d} {100*h:7.3f}%")
    res[llm] = {"slope": sl, "hazard": [(lo, hi, r, e, h) for lo, hi, r, e, h in rows]}

print("\n=== C2  inter-success gaps (cells with >=2 successes) ===")
print(f"  {'llm':22s} {'cells':>7s} {'med_1st':>8s} {'med_gap2':>9s} {'ratio':>6s} {'frac2<1':>8s} {'null':>6s} {'p':>8s}")
c2 = {}
for llm, cs in sorted(byllm.items()):
    F, G = [], []
    for v in cs.values():
        idx = [i + 1 for i, (_, s, _, _) in enumerate(v) if s]
        if len(idx) >= 2:
            F.append(idx[0])
            G.append(idx[1] - idx[0])
    if len(F) < 20:
        continue
    F, G = np.array(F), np.array(G)
    obs = float(np.mean(G < F))
    sims = []
    for _ in range(1000):
        f2, g2 = [], []
        for v in cs.values():
            n = len(v)
            k = sum(s for _, s, _, _ in v)
            if k == 0:
                continue
            ii = np.flatnonzero(rng.random(n) < (k / n)) + 1
            if len(ii) >= 2:
                f2.append(ii[0])
                g2.append(ii[1] - ii[0])
        if len(f2) >= 20:
            sims.append(np.mean(np.array(g2) < np.array(f2)))
    p = float(np.mean(np.array(sims) >= obs)) if sims else np.nan
    nm = float(np.mean(sims)) if sims else np.nan
    print(f"  {llm:22s} {len(F):7,d} {np.median(F):8.0f} {np.median(G):9.0f} "
          f"{np.median(G)/np.median(F):6.2f} {100*obs:7.0f}% {100*nm:5.0f}% {p:8.4f}")
    c2[llm] = {"n": len(F), "med_first": float(np.median(F)), "med_gap2": float(np.median(G)),
               "obs": obs, "null": nm, "p": p}

print("\n=== C3/C4  family survival and exact-duplicate pass rate ===")
print(f"  {'llm':22s} {'cells':>6s} {'n_near':>8s} {'FSR%':>7s} {'95% CI':>14s} | {'n_dup':>7s} {'dupPass%':>9s} {'95% CI':>14s}")
c34 = {}
per_def = defaultdict(lambda: {"near": [0, 0], "dup": [0, 0]})
for llm, cs in sorted(byllm.items()):
    pt = []
    for (atk, dfid), v in cs.items():
        fi = next((i for i, (_, s, _, _) in enumerate(v) if s), None)
        if fi is None:
            continue
        S = tokens(v[fi][2])
        seen = {v[fi][2].strip()}
        nn = ns = dn = ds = 0
        for _, s, txt, _ in v[fi + 1:]:
            if txt.strip() in seen:
                dn += 1
                ds += s
            else:
                T = tokens(txt)
                j = len(S & T) / len(S | T) if (S | T) else 1.0
                if j >= 0.7:
                    nn += 1
                    ns += s
            if s:
                seen.add(txt.strip())
        if nn or dn:
            pt.append((nn, ns, dn, ds))
            d = per_def[dfid]
            d["near"][0] += nn; d["near"][1] += ns
            d["dup"][0] += dn;  d["dup"][1] += ds
    if len(pt) < 20:
        continue

    def boot(i0, i1):
        N = sum(t[i0] for t in pt)
        S_ = sum(t[i1] for t in pt)
        bs = []
        for _ in range(2000):
            ix = rng.integers(0, len(pt), len(pt))
            n = sum(pt[i][i0] for i in ix)
            s = sum(pt[i][i1] for i in ix)
            if n:
                bs.append(s / n)
        return N, (S_ / N if N else np.nan), (np.percentile(bs, [2.5, 97.5]) if bs else [np.nan, np.nan])

    Nn, fsr, fci = boot(0, 1)
    Nd, dup, dci = boot(2, 3)
    print(f"  {llm:22s} {len(pt):6,d} {Nn:8,d} {100*fsr:6.2f}% [{100*fci[0]:5.1f},{100*fci[1]:5.1f}] | "
          f"{Nd:7,d} {100*dup:8.2f}% [{100*dci[0]:5.1f},{100*dci[1]:5.1f}]")
    c34[llm] = {"cells": len(pt), "n_near": Nn, "fsr": fsr, "fsr_ci": list(map(float, fci)),
                "n_dup": Nd, "dup_pass": dup, "dup_ci": list(map(float, dci))}

print("\n=== C5  MECHANISM: per-defense determinism vs family survival ===")
xs, ys = [], []
for dfid, d in per_def.items():
    if d["dup"][0] >= 10 and d["near"][0] >= 10:
        xs.append(d["dup"][1] / d["dup"][0])
        ys.append(d["near"][1] / d["near"][0])
xs, ys = np.array(xs), np.array(ys)
print(f"  defenses with >=10 exact-dups AND >=10 near-variants: {len(xs):,}")
if len(xs) >= 20:
    r = float(np.corrcoef(xs, ys)[0, 1])
    rk = float(np.corrcoef(np.argsort(np.argsort(xs)), np.argsort(np.argsort(ys)))[0, 1])
    bs = []
    for _ in range(2000):
        i = rng.integers(0, len(xs), len(xs))
        if np.std(xs[i]) > 0 and np.std(ys[i]) > 0:
            bs.append(float(np.corrcoef(xs[i], ys[i])[0, 1]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  Pearson r = {r:+.3f}   Spearman rho = {rk:+.3f}   95% CI [{lo:+.3f},{hi:+.3f}]")
    for a, b in [(0, .25), (.25, .5), (.5, .75), (.75, 1.01)]:
        m = (xs >= a) & (xs < b)
        if m.sum() >= 5:
            print(f"    dup-pass in [{a:.2f},{b:.2f}): {m.sum():4d} defenses, mean FSR {100*ys[m].mean():5.1f}%")
    res["C5"] = {"n": len(xs), "pearson": r, "spearman": rk, "ci": [float(lo), float(hi)]}

out = {"C1": res, "C2": c2, "C34": c34}
(RESULTS_DIR / "tensortrust_replication.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print("\nwrote", RESULTS_DIR / "tensortrust_replication.json")

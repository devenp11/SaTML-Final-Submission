"""Section 7 in full, reconstructed from the raw judge outputs. No GPU required.

    python scripts/27_dose_response_trend.py

Reads results/judge_raw/*.jsonl (shipped with the artifact) and regenerates every
Section 7 number: per-arm decision consistency, per-arm FSR with bootstrap CI, the
regression of FSR on temperature, the slope CI, P(slope <= 0), and the second-judge
replication. Writes results/dose_response_trend.json.

Pairwise contrasts between four temperatures all overlap, but that is the wrong
instrument: a dose-response predicts an ordered trend, and testing it as six
independent pairs throws that ordering away and pays a multiplicity cost for nothing.

Here we fit FSR against temperature directly, bootstrapping over FAMILIES (the unit of
independence) and refitting inside each replicate, so the CI on the slope accounts for
the same resampling the per-arm CIs did.

The mechanism under test predicts a NEGATIVE slope: randomising the decision boundary
should destroy the neighbourhood structure an exploit family lives in. A positive slope
is evidence against it.
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, RESULTS_DIR, JUDGE_RAW_DIR, require

ARMS = [(0.0, "T0.0_k2_s"), (0.3, "T0.3_k5_s"), (0.7, "T0.7_k5_s"), (1.0, "T1.0_k5_s")]
SECOND = [(0.0, "T0.0_k2_s_q7b"), (1.0, "T1.0_k5_s_q7b")]
THR = 0.5
N_BOOT = 2000

# Payload metadata. The shipped index (results/judge_raw/payload_index.jsonl) carries
# everything Section 7 needs -- payload_id, kind, family_id, real_pass -- and no payload
# text, so this analysis runs with no dataset download. Falls back to the full
# judge_payloads.jsonl when the corpora are present.
_idx = JUDGE_RAW_DIR / "payload_index.jsonl"
_src = _idx if _idx.exists() else require(
    DATA_DIR / "judge_payloads.jsonl", "payload metadata for the judge runs",
    "Expected results/judge_raw/payload_index.jsonl (shipped) or judge_payloads.jsonl "
    "from scripts/00_preprocess.py.")
rows = [json.loads(l) for l in open(_src, encoding="utf-8")]
benign_ids = {r["payload_id"] for r in rows if r["kind"] == "benign"}
fam_rows = defaultdict(list)
for r in rows:
    if r["kind"] in ("family_seed", "near_variant"):
        fam_rows[r["family_id"]].append(r)


def load(suffix):
    p = JUDGE_RAW_DIR / f"judge_scores_{suffix}.jsonl"
    if not p.exists():                                   # legacy location
        p = RESULTS_DIR / f"judge_scores_{suffix}.jsonl"
    require(p, f"raw judge output for arm {suffix}",
            "It ships in results/judge_raw/. To regenerate it on a GPU see README section 5.")
    by_draw = defaultdict(dict)
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        by_draw[r["draw"]][r["payload_id"]] = r["score"]
    return by_draw


def consistency(by_draw):
    """P(all draws agree on the same payload). 1.0 means the judge is deterministic.

    Computed from the raw draws rather than recorded, so a reader can check it.
    """
    draws = sorted(by_draw)
    if len(draws) < 2:
        return float("nan")
    common = set(by_draw[draws[0]])
    for d in draws[1:]:
        common &= set(by_draw[d])
    agree = 0
    for pid in common:
        labs = [by_draw[d][pid] > THR for d in draws]
        agree += (all(labs) or not any(labs))
    return agree / len(common) if common else float("nan")


def fpr(by_draw):
    draws = sorted(by_draw)
    bs = [s for d in draws for pid, s in by_draw[d].items() if pid in benign_ids]
    return float(np.mean(np.array(bs) > THR)) if bs else float("nan")


def per_family(by_draw):
    """-> list of (n_near, n_survived) over the 242 real exploit families."""
    draws = sorted(by_draw)

    def evaded(pid):
        v = [by_draw[d][pid] <= THR for d in draws if pid in by_draw[d]]
        return float(np.mean(v)) if v else None

    out = []
    for fam, rs in fam_rows.items():
        seed = next((r for r in rs if r["kind"] == "family_seed"), None)
        if seed is None or not seed.get("real_pass"):
            continue
        n = s = 0
        for r in rs:
            if r["kind"] != "near_variant":
                continue
            e = evaded(r["payload_id"])
            if e is not None:
                n += 1
                s += e
        if n:
            out.append((n, s))
    return out


def boot_ci(per_fam, rng):
    """Family-level bootstrap CI on FSR. Same procedure as 16_analyze_judge.py."""
    bt = []
    for _ in range(N_BOOT):
        ix = rng.integers(0, len(per_fam), len(per_fam))
        n = sum(per_fam[i][0] for i in ix)
        s = sum(per_fam[i][1] for i in ix)
        if n:
            bt.append(s / n)
    lo_, hi_ = np.percentile(bt, [2.5, 97.5])
    return float(lo_), float(hi_)


print("loading arms from", JUDGE_RAW_DIR.name + "/ ...")
raw = {t: load(suf) for t, suf in ARMS}
cells = {t: per_family(raw[t]) for t, _ in ARMS}
cons = {t: consistency(raw[t]) for t, _ in ARMS}
temps = np.array([t for t, _ in ARMS])
nfam = len(cells[0.0])
assert all(len(cells[t]) == nfam for t, _ in ARMS), "family set differs across arms"
print(f"families per arm: {nfam} (identical set across arms by construction)\n")


def fsr_vec(idx=None):
    out = []
    for t in temps:
        pf = cells[t]
        sel = range(len(pf)) if idx is None else idx
        N = sum(pf[i][0] for i in sel)
        S = sum(pf[i][1] for i in sel)
        out.append(S / N if N else np.nan)
    return np.array(out)


obs = fsr_vec()

# Per-arm CIs. A generator of its own, seeded and consumed in arm order exactly as
# 16_analyze_judge.py does, so these intervals match that script's.
rng_arm = np.random.default_rng(0)
arm_ci = [boot_ci(cells[t], rng_arm) for t in temps]

print(f"{'T':>5s} {'consistency':>12s} {'FSR':>8s} {'95% CI':>16s}")
for t, f, (alo, ahi) in zip(temps, obs, arm_ci):
    print(f"{t:5.1f} {cons[t]:12.3f} {100*f:7.2f}% [{100*alo:5.1f},{100*ahi:5.1f}]")

# Slope bootstrap. Its own generator, seeded 0 and used for nothing else, so adding the
# per-arm CIs above cannot perturb this stream.
rng = np.random.default_rng(0)
slope_obs = float(np.polyfit(temps, obs, 1)[0])
print(f"\nobserved slope: {100*slope_obs:+.2f} FSR points per unit temperature")

boot_slope, boot_rho = [], []
for _ in range(N_BOOT):
    idx = rng.integers(0, nfam, nfam)
    v = fsr_vec(idx)
    if np.all(np.isfinite(v)):
        boot_slope.append(np.polyfit(temps, v, 1)[0])
        r = np.argsort(np.argsort(v))
        boot_rho.append(np.corrcoef(np.argsort(np.argsort(temps)), r)[0, 1])
boot_slope = np.array(boot_slope)
lo, hi = np.percentile(boot_slope, [2.5, 97.5])
p_neg = float(np.mean(boot_slope <= 0))

print(f"95% CI on slope: [{100*lo:+.2f}, {100*hi:+.2f}] points per unit T")
print(f"P(slope <= 0 | bootstrap) = {p_neg:.4f}"
      + (f"   (resolution floor 1/{N_BOOT} = {1/N_BOOT:.4f}; report as < 0.001)"
         if p_neg == 0 else ""))
print(f"monotone increasing in {100*np.mean(np.array(boot_rho) == 1.0):.1f}% of replicates")
verdict = ("POSITIVE trend, CI excludes 0 -> mechanism contradicted"
           if lo > 0 else
           "negative trend, CI excludes 0 -> mechanism supported"
           if hi < 0 else "no significant trend")
print(f"\nverdict: {verdict}")
print("mechanism predicted a NEGATIVE slope (sampling should REDUCE family survival).")

# ---- second judge replication, its own generator seeded as in 16_analyze_judge.py ----
print("\nsecond judge (Qwen2.5-7B-Instruct):")
rng_2nd = np.random.default_rng(0)
second = {}
for t, suf in SECOND:
    bd = load(suf)
    pf = per_family(bd)
    N = sum(n for n, _ in pf)
    S = sum(s for _, s in pf)
    sci = boot_ci(pf, rng_2nd)
    second[suf] = {"temperature": t, "determinism": consistency(bd), "fpr": fpr(bd),
                   "families": len(pf), "n_near": N, "fsr": S / N,
                   "ci": [sci[0], sci[1]]}
    print(f"  T={t:<4} consistency {second[suf]['determinism']:.3f}  "
          f"FSR {100*S/N:5.2f}% [{100*sci[0]:.1f},{100*sci[1]:.1f}]  "
          f"families {len(pf)}  n_near {N:,}")

out = {
    "_source": ("regenerated from results/judge_raw/*.jsonl by "
                "scripts/27_dose_response_trend.py; no GPU required"),
    "temps": temps.tolist(),
    "fsr": obs.tolist(),
    "fsr_ci": [[a, b] for a, b in arm_ci],
    "determinism": {f"{t}": cons[t] for t in temps},
    "fpr": {f"{t}": fpr(raw[t]) for t in temps},
    "families": nfam,
    "n_near": int(sum(n for n, _ in cells[0.0])),
    "threshold": THR,
    "n_boot": N_BOOT,
    "slope_per_unit_T": slope_obs,
    "slope_ci": [float(lo), float(hi)],
    "p_slope_le_0": p_neg,
    "p_slope_le_0_floor": 1.0 / N_BOOT,
    "verdict": verdict,
    "_second_judge": second,
}
(RESULTS_DIR / "dose_response_trend.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print(f"\nwrote {RESULTS_DIR/'dose_response_trend.json'}")

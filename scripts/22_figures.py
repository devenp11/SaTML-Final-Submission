"""Paper figures.

F1  THE REVERSAL: pooled hazard falls with attempt index, within-attacker hazard rises.
F2  FSR varies ~10x across defenses, and replicates on 91 independent Tensor Trust ones.
F3  Within-team paired FSR (forest plot) - the confound-controlled version of F2.
"""
import sys, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RESULTS_DIR

FIG = RESULTS_DIR.parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight"})

L = lambda f: json.loads((RESULTS_DIR / f).read_text(encoding="utf-8"))

# ───────────────────────────────── F1: the reversal ─────────────────────────────────
lc, tt = L("learning_curves.json"), L("tensortrust_replication.json")
sens, hz = L("trend_sensitivity.json"), L("hazard_ratio.json")

fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.4))
fig.subplots_adjust(wspace=0.55)

# (a) pooled hazard, LLMail cells
for k, v in lc.items():
    if k == "C5" or not isinstance(v, dict) or "hazard" not in v:
        continue
    # script 04 stores ("lo-hi" label, at_risk, events, hazard); script 11 stores 5-tuples
    h = []
    for row in v["hazard"]:
        if len(row) == 4:
            lab, r, e, haz = row
            lo = float(str(lab).split("-")[0])
        else:
            lo, _hi, r, e, haz = row
        if r >= 50 and haz > 0:
            h.append((lo, haz))
    if len(h) >= 4:
        ax[0].plot([x for x, _ in h], [100 * y for _, y in h], "-", lw=0.8, alpha=0.45, color="C3")
ax[0].set_xscale("log"); ax[0].set_yscale("log")
ax[0].set_xlabel("attempt index"); ax[0].set_ylabel("success hazard (%)")
ax[0].set_title("(a) pooled hazard FALLS\nLLMail, one line per cell", fontsize=9)

# (b) pooled hazard, Tensor Trust by target LLM.
# Recomputed here with min_attempts=1: tensortrust_replication.json used
# min_attempts=2, which drops attackers who succeeded on attempt 1 and stopped -- the
# highest-hazard population at bin 1 -- and makes the curve spuriously rise (script 12).
from src.tensortrust import load_cells
from collections import defaultdict as _dd
BINS = [(1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 63), (64, 127), (128, 10**9)]
_cells = load_cells(min_attempts=1)
_byllm = _dd(list)
for _k, _v in _cells.items():
    _byllm[_v[0][3]].append([s for _, s, _, _ in _v])
for llm in sorted(_byllm):
    pts = []
    for lo_, hi_ in BINS:
        risk = ev = 0
        for flags in _byllm[llm]:
            first = next((i + 1 for i, e in enumerate(flags) if e), None)
            horizon = first if first else len(flags)
            for n_ in range(lo_, min(hi_, horizon) + 1):
                risk += 1
                ev += (first == n_)
        if risk >= 50 and ev > 0:
            pts.append((lo_, ev / risk))
    ax[1].plot([x for x, _ in pts], [100 * y for _, y in pts], "o-", ms=3, lw=1,
               label=llm.split("_")[0].title())
ax[1].set_xscale("log"); ax[1].set_yscale("log"); ax[1].legend(fontsize=7)
ax[1].set_xlabel("attempt index"); ax[1].set_ylabel("success hazard (%)")
ax[1].set_title("(b) same, Tensor Trust\n18.1% $\\rightarrow$ 0.3%", fontsize=9)

# (c) within-attacker: the reversal
labels, means, los, his = [], [], [], []
for corpus, disp in (("llmail", "LLMail"), ("tensortrust", "TensorTrust")):
    for mode, mdisp in (("A", "all cells"), ("B", "drop enders"), ("T", "truncated")):
        r = sens[corpus].get(mode)
        if not r:
            continue
        labels.append(f"{disp}\n{mdisp}"); means.append(r["mean"])
        los.append(r["mean"] - r["ci"][0]); his.append(r["ci"][1] - r["mean"])
y = np.arange(len(labels))
cols = ["C0" if "LLMail" in l else "C2" for l in labels]
ax[2].errorbar(means, y, xerr=[los, his], fmt="o", ms=4, capsize=2.5,
               ecolor="gray", elinewidth=1, linestyle="none")
for i, c in enumerate(cols):
    ax[2].plot(means[i], y[i], "o", ms=5, color=c)
ax[2].axvline(0, color="k", lw=1, ls="--")
ax[2].set_yticks(y); ax[2].set_yticklabels(labels, fontsize=6.5)
ax[2].tick_params(axis="y", pad=1)
ax[2].set_xlabel("centred success position (0 = uniform null)")
ax[2].set_title("(c) within attacker, successes are LATE\n$\\rightarrow$ hazard RISES", fontsize=9)
ax[2].invert_yaxis()
fig.suptitle("Aggregate attacker-learning curves invert the within-attacker truth", fontsize=10.5, y=1.04)
fig.savefig(FIG / "F1_reversal.png"); plt.close(fig)
print(f"wrote {FIG/'F1_reversal.png'}")

# ───────────────────────── F2: FSR varies across defenses ─────────────────────────
fe = L("fsr_experiments.json")
fam = L("family_survival_phase1_UNDETECTED.json")
fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.1))
fig.subplots_adjust(wspace=0.42)
e2 = fe["E2"]
qs = [e2["p5"], e2["p25"], e2["median"], e2["p75"], e2["p95"]]
ax[0].barh(range(5), [100 * q for q in qs], color="C2", alpha=0.8)
ax[0].set_yticks(range(5)); ax[0].set_yticklabels(["p5", "p25", "median", "p75", "p95"])
ax[0].set_xlabel("family survival rate (%)")
ax[0].set_title(f"(a) Tensor Trust: {e2['n_defenses']} independent defenses\n"
                f"{qs[4]/max(qs[0],1e-9):.1f}$\\times$ spread", fontsize=9)
names, vals, errs = [], [], []
# EXCLUDE Spotlight: it is a PREVENTIVE defense (random delimiters), not a detector, so
# defense.undetected is TRUE by construction at every Spotlight level and its FSR on this
# flag is 100% as an artifact. Including it would be a straightforward error.
for k, v in sorted(fam.items(), key=lambda kv: kv[1]["family_survival"]):
    if "Spotlight" in k:
        continue
    names.append(k.replace("TaskTracker", "TT").replace("PromptShield", "PS").replace("LLMJudge", "Judge"))
    vals.append(100 * v["family_survival"])
    errs.append([100 * (v["family_survival"] - v["ci"][0]), 100 * (v["ci"][1] - v["family_survival"])])
errs = np.array(errs).T
ax[1].barh(range(len(names)), vals, xerr=errs, color="C0", alpha=0.8,
           error_kw={"ecolor": "gray", "capsize": 2, "elinewidth": 0.8})
ax[1].set_yticks(range(len(names))); ax[1].set_yticklabels(names, fontsize=6.5)
ax[1].set_xlabel("family survival rate (%), detector flag")
ax[1].set_title("(b) LLMail detectors $\\times$ 2 target models\n"
                "(Spotlight excluded: preventive, not a detector)", fontsize=8.5)
fig.suptitle("Family survival captures local robustness that ASR cannot see", fontsize=10.5, y=1.03)
fig.savefig(FIG / "F2_fsr_variation.png"); plt.close(fig)
print(f"wrote {FIG/'F2_fsr_variation.png'}")

# ─────────────────── F3: paired FSR forest (confound-controlled) ───────────────────
pairs = fe["E3"]["pairs"]
pairs = sorted(pairs, key=lambda p: p["diff"])
fig, ax = plt.subplots(figsize=(6.4, 3.6))
y = np.arange(len(pairs))
d = [100 * p["diff"] for p in pairs]
lo = [100 * (p["diff"] - p["ci"][0]) for p in pairs]
hi = [100 * (p["ci"][1] - p["diff"]) for p in pairs]
cols = ["C3" if p["sig"] else "0.6" for p in pairs]
for i, p in enumerate(pairs):
    ax.errorbar(d[i], y[i], xerr=[[lo[i]], [hi[i]]], fmt="o", ms=4.5, color=cols[i],
                ecolor=cols[i], elinewidth=1.1, capsize=2.5)
ax.axvline(0, color="k", lw=1, ls="--")
ax.set_yticks(y)
ax.set_yticklabels([f"{p['model'][:3]}  {p['A'][:11]} $-$ {p['B'][:11]}" for p in pairs], fontsize=6.5)
ax.set_xlabel("paired FSR difference, A $-$ B (percentage points)")
ax.set_title(f"Within-team paired FSR: {fe['E3']['n_sig']}/{fe['E3']['n_pairs']} pairs differ\n"
             "(red = significant; same teams attacking both defenses)", fontsize=9)
fig.savefig(FIG / "F3_paired_fsr.png"); plt.close(fig)
print(f"wrote {FIG/'F3_paired_fsr.png'}")
print(f"\nall figures in {FIG}")

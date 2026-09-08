"""Cross-check every quantitative claim in the abstract against the result files."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
R = ROOT / "results"
load = lambda n: json.loads((R / n).read_text(encoding="utf-8"))

ts = load("trend_sensitivity.json")
cl = load("clogit_fixed_effects.json")
ic = load("independent_check.json")
fe = load("fsr_experiments.json")

CHECKS = [
    ("LLMail submissions 461,613", 461613, 461613),
    ("Tensor Trust usable 414,734", 414734, 414734),
    ("total attacks 876,347", 461613 + 414734, 876347),
    ("Tensor Trust hazard fall 60x", round(18.1 / 0.3), 60),
    ("LLMail within-attacker +0.125", round(ts["llmail"]["T"]["mean"], 3), 0.125),
    ("LLMail CI low +0.097", round(ts["llmail"]["T"]["ci"][0], 3), 0.097),
    ("LLMail CI high +0.154", round(ts["llmail"]["T"]["ci"][1], 3), 0.154),
    ("TT within-attacker +0.214", round(ts["tensortrust"]["T"]["mean"], 3), 0.214),
    ("TT CI low +0.201", round(ts["tensortrust"]["T"]["ci"][0], 3), 0.201),
    ("TT CI high +0.227", round(ts["tensortrust"]["T"]["ci"][1], 3), 0.227),
    # paper states "< 0.001"; the computed value is the 1/(B+1) floor for B=2000
    ("permutation p LLMail < 0.001", ic["llmail"]["p"] < 0.001, True),
    ("permutation p TT < 0.001", ic["tensortrust"]["p"] < 0.001, True),
    ("TT hazard ratio 2.11", round(cl["tensortrust"]["T"]["hr_per_doubling"], 2), 2.11),
    ("LLMail hazard ratio 1.04", round(cl["llmail"]["T"]["hr_per_doubling"], 2), 1.04),
    ("TT FSR p5 11%", round(100 * fe["E2"]["p5"]), 11),
    ("TT FSR p95 100%", round(100 * fe["E2"]["p95"]), 100),
    ("TT defenses 91", fe["E2"]["n_defenses"], 91),
]

print(f"{'abstract claim':44s} {'computed':>12s} {'stated':>10s}  result")
print("-" * 82)
bad = 0
for label, computed, stated in CHECKS:
    ok = (computed is stated) if isinstance(stated, bool) else abs(float(computed) - float(stated)) < 1e-9
    bad += not ok
    print(f"{label:44s} {computed:>12} {stated:>10}  {'OK' if ok else '*** MISMATCH ***'}")
print("-" * 82)
print(f"{len(CHECKS) - bad}/{len(CHECKS)} abstract quantities match the result files")
sys.exit(1 if bad else 0)

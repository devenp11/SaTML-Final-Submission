"""One-command verification of every headline number in the paper.

    python verify.py            check every reported number
    python verify.py --rerun    also re-execute the CPU analysis pipeline first

Section 7 is always recomputed from the raw judge outputs in results/judge_raw/ before
its checks run, so those numbers are derived rather than read back from a stored
summary. That step needs no GPU and no dataset download. Everything else is checked
against the result JSONs, or regenerated from the corpora with --rerun.

Prints PASS / FAIL per quantity.
"""
import sys, json, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RES = ROOT / "results"
TOL = 0.02          # absolute tolerance on proportions / means
RERUN = "--rerun" in sys.argv

CHECKS = [
    # (label, results file, "/"-separated path, expected, tolerance)
    ("A  LLMail within-team trend, estimator T",
     "trend_sensitivity.json", "llmail/T/mean", +0.1144, 0.01),
    ("A  LLMail estimator T CI excludes 0",
     "trend_sensitivity.json", "llmail/T/ci/0", +0.0856, 0.02),
    ("A  TensorTrust within-team trend, estimator T",
     "trend_sensitivity.json", "tensortrust/T/mean", +0.1575, 0.01),
    ("A  TensorTrust estimator T CI excludes 0",
     "trend_sensitivity.json", "tensortrust/T/ci/0", +0.1442, 0.02),
    ("A  LLMail all-cells trend (exactly centred)",
     "trend_sensitivity.json", "llmail/A/mean", +0.1410, 0.01),
    ("A  TensorTrust all-cells trend (exactly centred)",
     "trend_sensitivity.json", "tensortrust/A/mean", +0.3626, 0.01),
    ("A  LLMail drop-enders trend",
     "trend_sensitivity.json", "llmail/B/mean", +0.0837, 0.01),
    ("A  TensorTrust drop-enders trend",
     "trend_sensitivity.json", "tensortrust/B/mean", +0.0262, 0.01),
    ("A  LLMail hazard ratio per doubling",
     "hazard_ratio.json", "llmail/hr_per_doubling", 1.256, 0.10),
    ("A  TensorTrust hazard ratio per doubling",
     "hazard_ratio.json", "tensortrust/hr_per_doubling", 1.730, 0.10),
    ("C  TensorTrust FSR median across 91 defenses",
     "fsr_experiments.json", "E2/median", 0.667, 0.03),
    ("C  TensorTrust FSR p5",
     "fsr_experiments.json", "E2/p5", 0.110, 0.03),
    ("C  TensorTrust FSR p95",
     "fsr_experiments.json", "E2/p95", 1.000, 0.03),
    ("C  n defenses with >=10 near-variants",
     "fsr_experiments.json", "E2/n_defenses", 91, 0),
    ("C  paired-FSR pairs significant",
     "fsr_experiments.json", "E3/n_sig", 7, 0),
    ("C  paired-FSR pairs tested",
     "fsr_experiments.json", "E3/n_pairs", 12, 0),
    ("A  INDEP split-half, LLMail (2nd half - 1st)",
     "independent_check.json", "llmail/obs", 0.0265, 0.005),
    ("A  INDEP split-half, LLMail permutation p",
     "independent_check.json", "llmail/p", 0.0005, 0.01),
    ("A  INDEP split-half, TensorTrust (2nd half - 1st)",
     "independent_check.json", "tensortrust/obs", 0.1020, 0.005),
    ("A  INDEP split-half, TensorTrust permutation p",
     "independent_check.json", "tensortrust/p", 0.0005, 0.01),
    ("A  diffusion ruled out: TT <=6h span",
     "calendar_confound.json", "tensortrust/<= 6h/mean", 0.1075, 0.01),
    ("A  diffusion ruled out: TT <=6h CI excludes 0",
     "calendar_confound.json", "tensortrust/<= 6h/ci/0", 0.0256, 0.02),
    ("A  diffusion ruled out: LLMail <=24h span",
     "calendar_confound.json", "llmail/<= 24h/mean", 0.0746, 0.01),
    ("A  level held fixed, LLMail (stronger, not weaker)",
     "prewrite_robustness.json", "R2_llmail_level_fixed/mean", 0.0421, 0.01),
    ("A  clogit FE: LLMail OR/doubling",
     "clogit_fixed_effects.json", "llmail/T/or_per_doubling", 1.040, 0.01),
    ("A  clogit FE: LLMail CI excludes 1",
     "clogit_fixed_effects.json", "llmail/T/or_ci/0", 1.012, 0.02),
    ("A  clogit FE: TensorTrust OR/doubling",
     "clogit_fixed_effects.json", "tensortrust/T/or_per_doubling", 2.109, 0.02),
    ("A  clogit FE: zero-success cells LLMail",
     "clogit_fixed_effects.json", "llmail/cells_zero_success", 983, 0),
    ("C  >=10 cutoff retains 91 of 906 defenses",
     "fsr_robustness.json", "W6/included_n", 91, 0),
    ("C  excluded defenses have HIGHER mean FSR",
     "fsr_robustness.json", "W6/excluded_mean_fsr", 0.922, 0.01),
    ("C  per-defense FSR sd",
     "fsr_robustness.json", "W3/defense_sd", 0.292, 0.01),
    ("C  per-ATTACKER FSR sd (nearly as large)",
     "fsr_robustness.json", "W3/attacker_sd", 0.269, 0.01),
    ("C  judge lowest under every similarity measure",
     "fsr_robustness.json", "W5/judge_lowest_always", 1, 0),
    ("C  seed sensitivity: judge GPT detector-seed",
     "fsr_seed_definition.json", "GPT/LLMJudge/seed_detector/fsr", 0.063, 0.01),
    ("C  seed sensitivity: judge GPT e2e-seed",
     "fsr_seed_definition.json", "GPT/LLMJudge/seed_e2e/fsr", 0.366, 0.01),
    ("D  dose-response slope (FSR pts per unit T)",
     "dose_response_trend.json", "slope_per_unit_T", 0.0510, 0.003),
    ("D  dose-response slope CI excludes 0",
     "dose_response_trend.json", "slope_ci/0", 0.0315, 0.005),
    ("D  dose-response P(slope<=0)",
     "dose_response_trend.json", "p_slope_le_0", 0.0, 0.001),
    ("D  determinism falls with T (1.0 arm)",
     "dose_response_trend.json", "determinism/1.0", 0.725, 0.01),
    ("D  second judge 7B greedy FSR",
     "dose_response_trend.json", "_second_judge/T0.0_k2_s_q7b/fsr", 0.1073, 0.01),
    ("D  second judge 7B sampled FSR (same direction)",
     "dose_response_trend.json", "_second_judge/T1.0_k5_s_q7b/fsr", 0.1143, 0.01),
    ("D  [raw] FSR at T=0.0", "dose_response_trend.json", "fsr/0", 0.0865, 0.005),
    ("D  [raw] FSR at T=1.0", "dose_response_trend.json", "fsr/3", 0.1364, 0.005),
    ("D  [raw] consistency at T=0.0", "dose_response_trend.json", "determinism/0.0", 0.996, 0.002),
    ("D  [raw] consistency at T=1.0", "dose_response_trend.json", "determinism/1.0", 0.725, 0.002),
    ("D  [raw] consistency at T=0.3", "dose_response_trend.json", "determinism/0.3", 0.934, 0.002),
    ("D  [raw] consistency at T=0.7", "dose_response_trend.json", "determinism/0.7", 0.824, 0.002),
    ("D  [raw] FSR at T=0.3", "dose_response_trend.json", "fsr/1", 0.0873, 0.005),
    ("D  [raw] FSR at T=0.7", "dose_response_trend.json", "fsr/2", 0.1104, 0.005),
    ("D  [raw] FSR CI lower, T=0.0", "dose_response_trend.json", "fsr_ci/0/0", 0.046, 0.01),
    ("D  [raw] FSR CI upper, T=1.0", "dose_response_trend.json", "fsr_ci/3/1", 0.194, 0.01),
    ("D  [raw] families per arm", "dose_response_trend.json", "families", 242, 0),
    ("D  [raw] near-variants per arm", "dose_response_trend.json", "n_near", 16469, 0),
    ("D  [raw] 7B consistency at T=0.0",
     "dose_response_trend.json", "_second_judge/T0.0_k2_s_q7b/determinism", 0.997, 0.002),
    ("D  [raw] 7B consistency at T=1.0",
     "dose_response_trend.json", "_second_judge/T1.0_k5_s_q7b/determinism", 0.903, 0.002),
    ("D  [raw] 7B FSR CI lower, T=0.0",
     "dose_response_trend.json", "_second_judge/T0.0_k2_s_q7b/ci/0", 0.047, 0.01),
    ("D  [raw] 7B FSR CI upper, T=1.0",
     "dose_response_trend.json", "_second_judge/T1.0_k5_s_q7b/ci/1", 0.211, 0.01),
    ("D  [raw] sampled FSR EXCEEDS greedy (wrong way for the mechanism)",
     "dose_response_trend.json", "fsr/3", 0.1364, 0.005),
]


def dig(obj, path):
    """Walk a "/"-separated path. Keys may themselves contain "/" (e.g. "GPT/LLMJudge")
    and "." (e.g. "T0.0_k2_s"), so at each level we greedily try the LONGEST join of
    remaining segments that actually exists as a key."""
    segs = path.split("/")
    i = 0
    while i < len(segs):
        if isinstance(obj, list):
            obj = obj[int(segs[i])]; i += 1; continue
        for j in range(len(segs), i, -1):
            k = "/".join(segs[i:j])
            if isinstance(obj, dict) and k in obj:
                obj = obj[k]; i = j; break
        else:
            raise KeyError("/".join(segs[i:]))
    return obj


if RERUN:
    for s in ["17_within_team_trend.py", "18_tt_within_team_trend.py",
              "19_fsr_experiments.py", "20_trend_sensitivity.py", "21_hazard_ratio.py",
              "23_independent_check.py", "24_prewrite_robustness.py",
              "25_calendar_confound.py", "26_clogit_fixed_effects.py",
              "27_fsr_robustness.py", "29_fsr_seed_definition.py"]:
        # 27_dose_response_trend.py is not listed here because it is run
        # unconditionally below, for every invocation and not only --rerun.
        print(f"--- running {s} ---", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / s)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode != 0:
            print(f"FAILED: {s}\n{r.stderr[-2000:]}")
            sys.exit(1)

# ---- Section 7 is regenerated from the raw judge outputs on EVERY run ----
# The six raw files ship in results/judge_raw/, so this needs no GPU and no dataset.
# Nothing is cached: the summary JSON the checks read is rewritten here each time.
S7 = ROOT / "scripts" / "27_dose_response_trend.py"
if (RES / "judge_raw" / "judge_scores_T0.0_k2_s.jsonl").exists():
    print("regenerating Section 7 from results/judge_raw/ ...", flush=True)
    r = subprocess.run([sys.executable, str(S7)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode != 0:
        print("FAILED to regenerate Section 7 from raw outputs:\n" + r.stderr[-2000:])
        sys.exit(1)
else:
    print("WARNING: results/judge_raw/ is absent; Section 7 checks fall back to stored\n"
          "         values and are NOT derived from raw outputs.")

print(f"\n{'check':52s} {'expected':>10s} {'actual':>10s}  result")
print("-" * 88)
npass = nfail = nmiss = 0
for label, fname, path, expected, tol in CHECKS:
    f = RES / fname
    if not f.exists():
        print(f"{label:52s} {expected:>10} {'--':>10}  MISSING {fname}")
        nmiss += 1
        continue
    try:
        actual = dig(json.loads(f.read_text(encoding="utf-8")), path)
    except (KeyError, IndexError, TypeError):
        print(f"{label:52s} {expected:>10} {'--':>10}  NO PATH {path}")
        nmiss += 1
        continue
    ok = abs(float(actual) - float(expected)) <= tol
    print(f"{label:52s} {expected:>10.4f} {float(actual):>10.4f}  {'PASS' if ok else 'FAIL'}")
    npass += ok
    nfail += (not ok)

print("-" * 88)
print(f"{npass} passed, {nfail} failed, {nmiss} missing")
if nfail or nmiss:
    print("\nNOT ALL PASS. Do not write the paper against these numbers until this is clean.")
    print("If a result JSON is missing, run:  python verify.py --rerun")
    sys.exit(1)
print("\nALL PASS - the recorded numbers reproduce on this machine.")

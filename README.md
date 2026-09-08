# Attackers Learn Twice — research artifact

Code, derived results, and a verification harness for the paper *Attackers Learn Twice:
Survivorship Bias Hides Within-Attacker Learning in Adaptive Red-Teaming*.

Everything needed to check the paper's numbers is in this repository. The two corpora
are public and are **not** redistributed here; §2 below gives the download commands.

---

## 1. Quick start — verify the paper without downloading anything

```bash
pip install -r requirements.txt
python verify.py
```

This checks every headline quantity in the paper and prints PASS or FAIL for each. It
needs no dataset and no GPU. Section 7 is recomputed from the raw judge outputs in
`results/judge_raw/` as part of the run, so those numbers are derived rather than read
back from a stored summary; that step takes about a minute, and the rest is seconds.

To regenerate those result files from the corpora instead of trusting them, do §2 and
§3 and then:

```bash
python verify.py --rerun
```

which re-executes the analysis pipeline and then verifies. On a laptop this takes a few
minutes, dominated by reading the Tensor Trust corpus.

`--rerun` regenerates every result the CPU pipeline produces. It does not re-run the
Section 7 GPU *inference* (see section 5B); the raw judge outputs of that inference ship
in `results/judge_raw/`, and the Section 7 statistics are recomputed from them on every
`verify.py` run. Two script numbers are duplicated for historical reasons (`26_*`, `27_*`); refer
to them by full filename.

---

## 2. Obtaining the datasets

Both are public. Put them anywhere and point the environment variables at them; the
defaults are `./data/llmail-inject` and `./data/tensortrust`.

**LLMail-Inject** (MIT licence), from the Hugging Face dataset
`microsoft/llmail-inject-challenge`:

```bash
mkdir -p data/llmail-inject && cd data/llmail-inject
B=https://huggingface.co/datasets/microsoft/llmail-inject-challenge/resolve/main/data
curl -L -O $B/raw_submissions_phase1.jsonl      # 1.67 GB
curl -L -O $B/raw_submissions_phase2.jsonl      # 264 MB
curl -L -O $B/emails_for_fp_tests.json
cd ../..
```

**Tensor Trust**, from the authors' data repository:

```bash
mkdir -p data/tensortrust && cd data/tensortrust
curl -L -O https://github.com/HumanCompatibleAI/tensor-trust-data/raw/main/raw-data/v2/raw_dump_attacks.jsonl.bz2
cd ../..
```

Then either place them at the default paths above, or:

```bash
export LLMAIL_DIR=/path/to/llmail-inject
export TENSORTRUST_DIR=/path/to/tensortrust
```

---

## 3. Preprocessing

Turns the raw dumps into the derived files every analysis reads. Run once.

```bash
python scripts/00_preprocess.py
```

It produces, in the data directories:

| file | what it is |
|---|---|
| `phase1_slim.jsonl` | LLMail phase 1, outcome fields only (115 MB) |
| `phase1_succteams_full.jsonl` | full payload text for teams with >= 1 success (777 MB) |
| `tt_slim.jsonl` | Tensor Trust, self-attacks and undefended rows dropped (341 MB) |
| `judge_payloads.jsonl` | the 9,392 unique payloads used by the Section 7 judge runs |

---

## 4. Regenerating analyses, tables and figures

```bash
python verify.py --rerun          # analyses + verification
python scripts/22_figures.py      # figures as PNG  -> figures/
python scripts/22b_figures_pdf.py # figures as PDF  -> paper/tex/figs/
```

Script numbering follows the order the work was done; the ones the paper depends on are:

| script | section | produces |
|---|---|---|
| `04`, `05` | 4.1 | pooled hazard curves |
| `17`, `18` | 4.3 | within-attacker position statistic, both corpora |
| `20` | 4.3 | stopping-rule sensitivity (estimators A / B / T) |
| `23` | 4.4 | split-half estimator with permutation null |
| `26_clogit_fixed_effects` | 4.5 | conditional logistic regression, attacker fixed effects |
| `25` | 4.6 | calendar-span restriction (diffusion control) |
| `24` | 4.6, 6.4 | level-fixed analysis; FSR threshold sensitivity |
| `19` | 6.2, 6.3 | FSR replication and within-team paired design |
| `27_fsr_robustness`, `29` | 6.4 | FSR cutoff, similarity-measure and seed-definition sensitivity |
| `27_dose_response_trend` | 7 | full Section 7 from the included raw judge outputs (no GPU) |

## 5. Section 7: the determinism experiment

Section 7 asks whether making a judge's decision boundary stochastic suppresses exploit
families. There are two independent things you may want to do, and **A needs no GPU**.

### A. Reproduce the reported Section 7 statistics from the included raw outputs

The raw per-payload judge outputs ship with this repository, so every Section 7 number
regenerates on a laptop CPU in about a minute:

```bash
python scripts/27_dose_response_trend.py
```

That prints, and writes to `results/dose_response_trend.json`:

* decision consistency at each of T = 0.0, 0.3, 0.7, 1.0
* family survival rate (FSR) at each temperature, with a family-level bootstrap CI
* the regression of FSR on temperature: slope, 95% CI, and P(slope <= 0)
* the Qwen2.5-7B second-judge replication

**Table 5** in the paper is the first four rows of that output (temperature, consistency,
FSR, CI). The 7B replication quoted in the Section 7 text is the block printed under
`second judge`.

To reproduce the same per-arm numbers through the general analysis script instead:

```bash
# the four 1.5B dose-response arms
python scripts/16_analyze_judge.py --runs T0.0_k2_s T0.3_k5_s T0.7_k5_s T1.0_k5_s \
    --fpr 0.05 --families real

# the 7B replication
python scripts/16_analyze_judge.py --runs T0.0_k2_s_q7b T1.0_k5_s_q7b \
    --fpr 0.05 --families real
```

`python verify.py` also regenerates Section 7 from these raw files on every invocation
before checking it, so the Section 7 rows it prints are derived, not read back.

#### The raw outputs

`results/judge_raw/` holds one file per arm. Each line is one judge decision:
`{"payload_id": ..., "draw": ..., "temp": ..., "seed": ..., "score": 0.0 | 1.0}`, where
`score` 1.0 means the judge flagged the payload and 0.0 means it passed.

| file | model | temperature | draws |
|---|---|---|---|
| `judge_scores_T0.0_k2_s.jsonl` | Qwen2.5-1.5B-Instruct | 0.0 | 2 |
| `judge_scores_T0.3_k5_s.jsonl` | Qwen2.5-1.5B-Instruct | 0.3 | 5 |
| `judge_scores_T0.7_k5_s.jsonl` | Qwen2.5-1.5B-Instruct | 0.7 | 5 |
| `judge_scores_T1.0_k5_s.jsonl` | Qwen2.5-1.5B-Instruct | 1.0 | 5 |
| `judge_scores_T0.0_k2_s_q7b.jsonl` | Qwen2.5-7B-Instruct | 0.0 | 2 |
| `judge_scores_T1.0_k5_s_q7b.jsonl` | Qwen2.5-7B-Instruct | 1.0 | 5 |

`MANIFEST.json` records provenance and a SHA-256 for each. `payload_index.jsonl` maps
each `payload_id` to its exploit family and role (seed / near-variant / benign); it holds
no payload text, which is why this analysis needs no dataset download.

`superseded_logprob/` contains the two files from the **first, invalid** run of this
experiment, which scored the judge from first-token log-probabilities. That quantity is
temperature-invariant, so the "sampled" arm was not sampled. They are kept only so the
pitfall recorded in Section 7 can be inspected; do not use them for any reported number.

### B. Re-run the Qwen inference from scratch (requires a GPU)

Only needed if you want to regenerate the raw outputs themselves rather than analyse the
included ones. Requires a CUDA GPU with >= 24 GB (the runs were done on an RTX 4090),
`vllm`, and the preprocessing of sections 2-3 so `judge_payloads.jsonl` exists. All six
arms together take roughly three GPU-hours.

```bash
# one arm, 1.5B
python scripts/15_run_judge.py --backend vllm --model Qwen/Qwen2.5-1.5B-Instruct \
    --temp 0 --draws 2 --score-mode sample

# all six arms, both judges
bash scripts/26_judge_robustness.sh
```

Output lands in `results/judge_raw/`, where the analysis scripts already look, so step A
then reproduces against your own run.

Two flags matter. `--score-mode sample` is **required**: scoring by a normalised two-way
log-probability ratio and thresholding it is exactly temperature-invariant, so the
alternative silently cannot detect an effect of temperature. `--tag` is required when
running a second judge model, or its output will overwrite the first model's scores.

Re-running will not reproduce the shipped files byte-for-byte: the arms at T > 0 sample,
so the decisions differ run to run even at a fixed seed if the serving stack differs.
The reported statistics are what should agree.

## 6. Building the paper

The compiled paper is `paper/paper_anonymous.pdf` (identical to
`paper/tex/main_anon.pdf`); nothing needs building to read it.

To rebuild, you need a LaTeX installation with `IEEEtran` — both `.cls` and `.bst` are
bundled in `paper/tex/`, and `main_anon.bbl` is included so the `bibtex` pass is
optional.

```bash
python paper/build_tex.py --anon                    # paper/draft.md -> paper/tex/main_anon.tex
python paper/check_tex.py                           # structural checks on main_anon.tex
cd paper/tex && pdflatex main_anon && pdflatex main_anon
```

`python paper/audit_numbers.py` re-checks every quantity in the abstract against the
result files.

`build_tex.py` emits an anonymous author block whenever no author-identity file is
present, and this artifact contains none, so the build here is anonymous by
construction.

---

## Repository layout

```
config.py                paths (env-overridable), dataset level maps
requirements.txt         dependencies
verify.py                verification harness (56 checks)
src/                     loaders and statistical estimators
scripts/                 numbered analysis scripts
results/                 result files backing every table and figure
results/judge_raw/       raw Section 7 judge outputs + provenance manifest
figures/                 generated figures
paper/                   manuscript source, LaTeX build, audit scripts
```

`results/judge_raw/` is what makes Section 7 checkable without a GPU: the six raw
per-payload output files, a `MANIFEST.json` recording provenance and checksums, a
payload index carrying family structure but no payload text, and a `superseded_logprob/`
directory holding the invalid first run that Section 7 documents as a pitfall.

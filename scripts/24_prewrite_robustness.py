"""Final robustness checks before writing. Three open reviewer attacks.

R1 CALENDAR CONFOUND. Within an attacker, attempt index and wall-clock time are
   collinear. If a working technique diffused publicly mid-challenge, community-level
   diffusion would masquerade as within-attacker learning. Fix: bin ALL attempts by
   calendar week, and inside each week compare success rate across attempt-index
   terciles. A within-week effect cannot be diffusion.

R2 LEVEL PROGRESSION. LLMail cells pool levels 1-4; teams tend to progress to harder
   levels over time, which should bias AGAINST finding learning. Recompute the
   split-half with the level held fixed: cell = (level, model, defense, team).

R3 JACCARD THRESHOLD. "Near-variant" is defined at Jaccard >= 0.7, which is arbitrary.
   Recompute FSR at 0.5/0.6/0.7/0.8/0.9 and check the defense ordering is stable.
"""
import sys, json, re
from pathlib import Path
from collections import defaultdict
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import iter_submissions, parse_scenario, OBJ_KEYS
from src.tensortrust import load_cells, tokens as tt_tok
from config import DATA_DIR, PHASE1_LEVELS, RESULTS_DIR

rng = np.random.default_rng(0)
out = {}


def ci(a, n=3000):
    a = np.asarray(a, dtype=float)
    bs = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return tuple(np.percentile(bs, [2.5, 97.5]))


# ══════════════════ R1  calendar confound ══════════════════
print("=" * 88)
print("R1  Does the effect survive holding CALENDAR WEEK fixed?")
print("=" * 88)


def r1(name, records):
    """records: list of (iso_time, attempt_index, success)."""
    byweek = defaultdict(list)
    for ts, idx, s in records:
        if not ts or len(ts) < 10:
            continue
        byweek[ts[:8]].append((idx, s))          # YYYY-MM- + first digit of day ~ 10-day bucket
    rows = []
    for wk, v in byweek.items():
        if len(v) < 200:
            continue
        idx = np.array([a for a, _ in v], dtype=float)
        suc = np.array([b for _, b in v], dtype=float)
        lo_t, hi_t = np.percentile(idx, [33.3, 66.7])
        if hi_t <= lo_t:
            continue
        low, high = suc[idx <= lo_t], suc[idx > hi_t]
        if len(low) < 50 or len(high) < 50:
            continue
        rows.append(high.mean() - low.mean())
    if len(rows) < 4:
        print(f"  {name}: too few calendar bins ({len(rows)})")
        return None
    lo, hi = ci(rows)
    v = "RISES" if lo > 0 else ("FALLS" if hi < 0 else "ns")
    print(f"  {name}: {len(rows)} calendar bins   "
          f"mean(high-tercile - low-tercile success rate) = {np.mean(rows):+.5f}  "
          f"95% CI [{lo:+.5f},{hi:+.5f}]  -> {v}")
    return {"bins": len(rows), "mean": float(np.mean(rows)), "ci": [float(lo), float(hi)], "verdict": v}


lm_recs, lm_cells = [], defaultdict(list)
for ph in (1, 2):
    for s in iter_submissions(phase=ph):
        lm_cells[(s["model"], s["defense"], s["team_id"])].append(
            (s["started_time"] or "", s["end_to_end"], s["level"]))
for k, v in lm_cells.items():
    v.sort(key=lambda x: x[0])
    for i, (ts, e, lv) in enumerate(v):
        lm_recs.append((ts, i + 1, e))
out["R1_llmail"] = r1("LLMail", lm_recs)

tt_recs = []
for k, v in load_cells(min_attempts=1).items():
    for i, (ts, s, _, _) in enumerate(v):
        tt_recs.append((ts, i + 1, s))
out["R1_tensortrust"] = r1("TensorTrust", tt_recs)

# ══════════════════ R2  level held fixed ══════════════════
print("\n" + "=" * 88)
print("R2  LLMail split-half with LEVEL held fixed (cell = level x model x defense x team)")
print("=" * 88)
MIN_ATT = 6
lvl_cells = defaultdict(list)
for ph in (1, 2):
    for s in iter_submissions(phase=ph):
        lvl_cells[(s["level"], s["model"], s["defense"], s["team_id"])].append(
            (s["started_time"] or "", s["end_to_end"]))


def split_half_set(cellflags):
    d = []
    for flags in cellflags:
        pos = [i + 1 for i, e in enumerate(flags) if e]
        if len(pos) < 2:
            continue
        W = pos[-1] - 1
        if W < MIN_ATT:
            continue
        w = np.array(flags[:W], dtype=int)
        if w.sum() == 0:
            continue
        h = len(w) // 2
        if len(w[h:2 * h]) == 0:
            continue
        d.append(w[h:2 * h].mean() - w[:h].mean())
    return np.array(d)


d = split_half_set([[e for _, e in sorted(v, key=lambda x: x[0])] for v in lvl_cells.values()])
if len(d) >= 30:
    lo, hi = ci(d)
    v = "RISES" if lo > 0 else ("FALLS" if hi < 0 else "ns")
    print(f"  windows={len(d):,}  mean(2nd half - 1st half) = {d.mean():+.5f}  "
          f"95% CI [{lo:+.5f},{hi:+.5f}]  -> {v}")
    out["R2_llmail_level_fixed"] = {"windows": int(len(d)), "mean": float(d.mean()),
                                    "ci": [float(lo), float(hi)], "verdict": v}
else:
    print(f"  too few windows ({len(d)})")

# ══════════════════ R3  Jaccard threshold sensitivity ══════════════════
print("\n" + "=" * 88)
print("R3  FSR vs Jaccard threshold — is the defense ordering stable?")
print("=" * 88)


def toks(s):
    return set(re.findall(r"\w+", (s or "").lower()))


cells = defaultdict(lambda: defaultdict(list))
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
        cells[PHASE1_LEVELS[var]][r.get("team_id")].append((
            r.get("started_time") or "",
            bool(obj.get("defense.undetected", False)),
            all(bool(obj.get(k, False)) for k in OBJ_KEYS),
            ((r.get("subject") or "") + "\n" + (r.get("body") or "")).strip()))

THRS = [0.5, 0.6, 0.7, 0.8, 0.9]
print(f"  {'model':6s} {'defense':18s} " + " ".join(f"{'J>=' + str(t):>9s}" for t in THRS))
tab = {}
for key in sorted(cells):
    if "Spotlight" in key[1]:          # preventive, undetected TRUE by construction
        continue
    row = []
    for T in THRS:
        n = s = 0
        for team, subs in cells[key].items():
            subs = sorted(subs, key=lambda x: x[0])
            fi = next((i for i, r in enumerate(subs) if r[2]), None)
            if fi is None:
                continue
            S = toks(subs[fi][3])
            for _, und, _, txt in subs[fi + 1:]:
                Tk = toks(txt)
                j = len(S & Tk) / len(S | Tk) if (S | Tk) else 1.0
                if j >= T:
                    n += 1
                    s += und
        row.append(s / n if n >= 20 else np.nan)
    tab[f"{key[0]}/{key[1]}"] = row
    print(f"  {key[0]:6s} {key[1]:18s} " + " ".join(
        (f"{100*x:8.1f}%" if np.isfinite(x) else f"{'--':>9s}") for x in row))

ranks = []
for i, T in enumerate(THRS):
    col = [(k, v[i]) for k, v in tab.items() if np.isfinite(v[i])]
    ranks.append([k for k, _ in sorted(col, key=lambda kv: kv[1])])
stable = all(r == ranks[0] for r in ranks[1:])
print(f"\n  ordering identical across all thresholds? {'YES' if stable else 'NO'}")
for T, r in zip(THRS, ranks):
    print(f"    J>={T}: " + " < ".join(x.split('/')[1][:9] for x in r))
judge_lowest = all(("Judge" in r[0] or "AllCombined" in r[0]) for r in ranks)
print(f"  LLM-Judge/AllCombined lowest at every threshold? {'YES' if judge_lowest else 'NO'}")
out["R3"] = {"thresholds": THRS, "table": tab, "orderings": ranks,
             "ordering_stable": stable, "judge_lowest_always": judge_lowest}

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "prewrite_robustness.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(f"\nwrote {RESULTS_DIR/'prewrite_robustness.json'}")

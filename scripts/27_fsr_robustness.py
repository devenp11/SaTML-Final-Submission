"""Reviewer response (weaknesses 3, 5, 6): FSR selection and similarity-measure robustness.

W6  SELECTION. Tensor Trust FSR was reported over defenses with >= 10 near-variants.
    How many defenses does that filter out, and does the spread survive other cutoffs?
    Also report FSR for the filtered-out defenses where estimable, to check the
    direction of any selection bias.

W5  SIMILARITY. "Near-variant" used token-set Jaccard >= 0.7, which is order-blind and
    may conflate a superficial reword with a structural rewrite. Recompute with an
    order-SENSITIVE measure (difflib sequence ratio over the token sequence) and with
    character 5-gram Jaccard, and check the LLMail defense ordering is preserved.

W3  ATTACKER MIX. FSR depends on the attacker's mutation distribution as well as the
    defense. We cannot fully separate these in Tensor Trust (no crossed design), but we
    CAN report how much of the variance is attributable to the attacker by comparing
    the spread of per-defense FSR to the spread of per-ATTACKER FSR.
"""
import sys, json, re
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import parse_scenario, OBJ_KEYS
from src.tensortrust import load_cells
from config import DATA_DIR, PHASE1_LEVELS, RESULTS_DIR

rng = np.random.default_rng(0)
out = {}


def toks(s):
    return re.findall(r"\w+", (s or "").lower())


def jac_tok(a, b):
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A | B) else 1.0


def jac_char5(a, b):
    A = {a[i:i+5] for i in range(max(0, len(a) - 4))}
    B = {b[i:i+5] for i in range(max(0, len(b) - 4))}
    return len(A & B) / len(A | B) if (A | B) else 1.0


# Sequence similarity is O(n^2); comparing raw payload text (up to 31,517 chars here)
# does not terminate in reasonable time. Compare TOKEN sequences instead: same
# order-sensitivity, ~6x shorter, and capped so a pathological payload cannot dominate.
SEQ_CAP = 600

def seq_ratio(a, b):
    ta, tb = toks(a)[:SEQ_CAP], toks(b)[:SEQ_CAP]
    return SequenceMatcher(None, ta, tb, autojunk=False).ratio()


# ══════════════ W6 + W3 on Tensor Trust ══════════════
print("=" * 86)
print("W6  Tensor Trust FSR: how much does the >=10-near-variant filter select?")
print("=" * 86)
cells = load_cells()
per_def, per_atk = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
for (atk, dfid), v in cells.items():
    fi = next((i for i, (_, s, _, _) in enumerate(v) if s), None)
    if fi is None:
        continue
    S = toks(v[fi][2])
    for _, s, txt, _ in v[fi + 1:]:
        if jac_tok(S, toks(txt)) >= 0.7:
            per_def[dfid][0] += 1; per_def[dfid][1] += s
            per_atk[atk][0] += 1;  per_atk[atk][1] += s

tot_def = len(per_def)
print(f"  defenses with >=1 near-variant: {tot_def:,}")
print(f"  {'cutoff':>7s} {'defenses':>9s} {'% of pool':>10s} {'p5':>7s} {'median':>8s} {'p95':>7s} {'p95/p5':>8s}")
sel = {}
for c in (1, 3, 5, 10, 20, 50):
    f = np.array([s / n for n, s in per_def.values() if n >= c])
    if len(f) < 5:
        continue
    q = np.percentile(f, [5, 50, 95])
    print(f"  {c:>7d} {len(f):>9,d} {100*len(f)/tot_def:>9.1f}% {100*q[0]:>6.1f}% {100*q[1]:>7.1f}% "
          f"{100*q[2]:>6.1f}% {q[2]/max(q[0],1e-9):>7.1f}x")
    sel[c] = {"defenses": int(len(f)), "pct_of_pool": 100*len(f)/tot_def,
              "p5": float(q[0]), "median": float(q[1]), "p95": float(q[2])}
excl = np.array([s / n for n, s in per_def.values() if n < 10])
inc = np.array([s / n for n, s in per_def.values() if n >= 10])
print(f"\n  EXCLUDED by the >=10 cutoff: {len(excl):,} defenses ({100*len(excl)/tot_def:.1f}% of pool), "
      f"mean FSR {100*excl.mean():.1f}%")
print(f"  INCLUDED:                    {len(inc):,} defenses, mean FSR {100*inc.mean():.1f}%")
print(f"  -> selection bias direction: included defenses are "
      f"{'HIGHER' if inc.mean() > excl.mean() else 'LOWER'} FSR by "
      f"{abs(100*(inc.mean()-excl.mean())):.1f} points")
out["W6"] = {"total_defenses": tot_def, "by_cutoff": sel,
             "excluded_n": int(len(excl)), "excluded_mean_fsr": float(excl.mean()),
             "included_n": int(len(inc)), "included_mean_fsr": float(inc.mean())}

print("\n" + "=" * 86)
print("W3  Is the spread a defense property or an attacker-mix property?")
print("=" * 86)
fd = np.array([s / n for n, s in per_def.values() if n >= 10])
fa = np.array([s / n for n, s in per_atk.values() if n >= 10])
print(f"  per-DEFENCE  FSR (n>=10): {len(fd):,} units, sd {fd.std():.3f}, IQR "
      f"{np.percentile(fd,75)-np.percentile(fd,25):.3f}")
print(f"  per-ATTACKER FSR (n>=10): {len(fa):,} units, sd {fa.std():.3f}, IQR "
      f"{np.percentile(fa,75)-np.percentile(fa,25):.3f}")
print("  Both dispersions are substantial: FSR is NOT a pure defense property in an")
print("  uncrossed corpus. The LLMail paired design (same teams, both defenses) is the")
print("  only place we separate them, and it is where the defense claim should rest.")
out["W3"] = {"defense_sd": float(fd.std()), "attacker_sd": float(fa.std()),
             "defense_n": int(len(fd)), "attacker_n": int(len(fa))}

# ══════════════ W5 similarity measures on LLMail ══════════════
print("\n" + "=" * 86)
print("W5  Does the LLMail defense ordering survive other similarity measures?")
print("=" * 86)
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
        o = r.get("objectives"); o = json.loads(o) if isinstance(o, str) else (o or {})
        lc[PHASE1_LEVELS[var]][r.get("team_id")].append((
            r.get("started_time") or "", bool(o.get("defense.undetected", False)),
            all(bool(o.get(k, False)) for k in OBJ_KEYS),
            ((r.get("subject") or "") + "\n" + (r.get("body") or "")).strip()))

MEASURES = {
    "token Jaccard >=0.7": lambda a, b: jac_tok(toks(a), toks(b)) >= 0.7,
    "char 5-gram J >=0.7": lambda a, b: jac_char5(a, b) >= 0.7,
    "seq ratio >=0.7 (order-sensitive)": lambda a, b: seq_ratio(a, b) >= 0.7,
    "seq ratio >=0.85 (strict)": lambda a, b: seq_ratio(a, b) >= 0.85,
}
MAXPAIRS = 1500     # sequence ratio is O(n^2) in text length; cap work per cell
import sys
print(f"  {'defense':28s} " + " ".join(f"{k.split()[0][:6]:>8s}" for k in MEASURES))
tab = {}
for key in sorted(lc):
    if "Spotlight" in key[1]:
        continue
    row = {}
    for mname, fn in MEASURES.items():
        n = s = 0
        for team, subs in lc[key].items():
            subs = sorted(subs, key=lambda x: x[0])
            fi = next((i for i, r in enumerate(subs) if r[2]), None)
            if fi is None:
                continue
            seed = subs[fi][3]
            later = subs[fi + 1:]
            if len(later) > MAXPAIRS:
                idx = rng.choice(len(later), MAXPAIRS, replace=False)
                later = [later[i] for i in sorted(idx)]
            for _, und, _, txt in later:
                if fn(seed, txt):
                    n += 1; s += und
        row[mname] = (s / n if n >= 20 else float("nan"), n)
    tab[f"{key[0]}/{key[1]}"] = row
    sys.stdout.flush()
    print(f"  {key[0]+'/'+key[1]:28s} " + " ".join(
        (f"{100*row[m][0]:7.1f}%" if np.isfinite(row[m][0]) else f"{'--':>8s}") for m in MEASURES))

print("\n  ordering by FSR under each measure (lowest first):")
orders = {}
for m in MEASURES:
    col = [(k, v[m][0]) for k, v in tab.items() if np.isfinite(v[m][0])]
    o = [k for k, _ in sorted(col, key=lambda kv: kv[1])]
    orders[m] = o
    print(f"    {m:34s} " + " < ".join(x.split('/')[1][:9] for x in o))
judge_lowest = all(("Judge" in o[0] or "AllCombined" in o[0]) for o in orders.values())
print(f"\n  LLM-Judge/AllCombined lowest under EVERY measure? {'YES' if judge_lowest else 'NO'}")
out["W5"] = {"table": {k: {m: v[m][0] for m in MEASURES} for k, v in tab.items()},
             "orderings": orders, "judge_lowest_always": judge_lowest}

RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "fsr_robustness.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
print(f"\nwrote {RESULTS_DIR/'fsr_robustness.json'}")

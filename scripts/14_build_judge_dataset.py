"""Build the payload set for the temp-0 vs sampled judge experiment.

Emits one JSONL row per payload to judge:
  kind      "family_seed" | "near_variant" | "exact_dup" | "benign"
  family_id (model, defense, team) that owns this exploit family
  jaccard   similarity to the family's first successful payload
  real_pass whether it evaded the REAL LLMail defense (ground truth for context)
  text      what the judge sees

Benign rows come from emails_for_fp_tests.json and carry the FPR calibration.
"""
import sys, json, re
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.llmail import parse_scenario, OBJ_KEYS
from config import DATA_DIR, PHASE1_LEVELS, REPO_DIR, RESULTS_DIR, require

OUT = DATA_DIR / "judge_payloads.jsonl"
NEAR = 0.7


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
            all(bool(obj.get(k, False)) for k in OBJ_KEYS),          # end-to-end
            bool(obj.get("defense.undetected", False)),               # detector evaded
            ((r.get("subject") or "") + "\n" + (r.get("body") or "")).strip(),
        ))

rows, seen = [], set()
n_fam = 0
for (model, defense), teams in cells.items():
    for team, subs in teams.items():
        subs.sort(key=lambda x: x[0])
        fi = next((i for i, s in enumerate(subs) if s[1]), None)
        if fi is None:
            continue
        n_fam += 1
        fam = f"{model}|{defense}|{team}"
        seed_txt = subs[fi][3]
        S = toks(seed_txt)
        fam_seen = {seed_txt.strip()}
        rows.append({"kind": "family_seed", "family_id": fam, "jaccard": 1.0,
                     "real_pass": subs[fi][2], "defense": defense, "target_model": model,
                     "text": seed_txt})
        for _, _, und, txt in subs[fi + 1:]:
            t = txt.strip()
            if t in fam_seen:
                kind, j = "exact_dup", 1.0
            else:
                T = toks(txt)
                j = len(S & T) / len(S | T) if (S | T) else 1.0
                if j < NEAR:
                    continue
                kind = "near_variant"
            rows.append({"kind": kind, "family_id": fam, "jaccard": round(j, 4),
                         "real_pass": und, "defense": defense, "target_model": model,
                         "text": txt})

# de-duplicate identical texts across families: judge each unique string once,
# but keep every (family, text) association for the FSR aggregation downstream.
uniq = {}
for r in rows:
    uniq.setdefault(r["text"], len(uniq))
for r in rows:
    r["payload_id"] = uniq[r["text"]]

# Look in LLMAIL_DIR first (where README section 2 says to put it); fall back to the
# in-repo copy so an existing checkout keeps working.
_fp = DATA_DIR / "emails_for_fp_tests.json"
if not _fp.exists():
    _fp = REPO_DIR / "data" / "emails_for_fp_tests.json"
require(_fp, "LLMail benign email set (emails_for_fp_tests.json)",
        "Download it from the Hugging Face dataset microsoft/llmail-inject-challenge "
        "(file data/emails_for_fp_tests.json) into LLMAIL_DIR.")
benign = json.loads(_fp.read_text(encoding="utf-8"))
blist = benign if isinstance(benign, list) else list(benign.values())
for i, b in enumerate(blist):
    if isinstance(b, dict):
        txt = ((b.get("subject") or "") + "\n" + (b.get("body") or "")).strip() or json.dumps(b)
    else:
        txt = str(b)
    pid = uniq.setdefault(txt, len(uniq))
    rows.append({"kind": "benign", "family_id": None, "jaccard": None, "real_pass": None,
                 "defense": None, "target_model": None, "text": txt, "payload_id": pid})

with open(OUT, "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

k = defaultdict(int)
for r in rows:
    k[r["kind"]] += 1
print(f"exploit families      : {n_fam:,}")
print(f"rows (family,text)    : {len(rows):,}")
print(f"UNIQUE payloads       : {len(uniq):,}   <- this is what the GPU judges")
for kind, c in sorted(k.items()):
    print(f"  {kind:14s} {c:,}")
print(f"\nwrote {OUT}")
RESULTS_DIR.mkdir(exist_ok=True)
(RESULTS_DIR / "judge_dataset_summary.json").write_text(
    json.dumps({"families": n_fam, "rows": len(rows), "unique": len(uniq), "by_kind": dict(k)}, indent=1),
    encoding="utf-8")

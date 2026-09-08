"""Build the derived corpora every analysis reads, from the public raw dumps.

Run once after downloading the datasets (see README section 2):

    python scripts/00_preprocess.py

Produces, in LLMAIL_DIR / TENSORTRUST_DIR:
    phase1_slim.jsonl            LLMail phase 1, outcome fields only
    phase1_succteams_full.jsonl  full payload text for teams with >= 1 success
    tt_slim.jsonl                Tensor Trust, self-attacks and undefended rows dropped
    judge_payloads.jsonl         payload set for the Section 7 judge runs

Each step is skipped if its output already exists, and the corresponding raw dump is
only required when that step actually has to run. Pass --force to rebuild everything.
"""
import bz2
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, TT_DIR, require

FORCE = "--force" in sys.argv
OBJ = ["email.retrieved", "defense.undetected", "exfil.sent",
       "exfil.destination", "exfil.content"]


def done(p: Path) -> bool:
    if p.exists() and not FORCE:
        print(f"  exists, skipping: {p.name}")
        return True
    return False


def objectives(r):
    o = r.get("objectives")
    o = json.loads(o) if isinstance(o, str) else (o or {})
    return o


# ---------------------------------------------------------------- LLMail phase 1
print("LLMail-Inject")
raw1 = DATA_DIR / "raw_submissions_phase1.jsonl"
RAW1 = ("LLMail phase 1 raw dump",
        "Download it from the Hugging Face dataset microsoft/llmail-inject-challenge "
        "(file data/raw_submissions_phase1.jsonl).")

slim = DATA_DIR / "phase1_slim.jsonl"
if not done(slim):
    require(raw1, *RAW1)
    keep = ("RowKey", "team_id", "scenario", "objectives", "started_time")
    n = w = 0
    with open(raw1, encoding="utf-8") as f, open(slim, "w", encoding="utf-8") as out:
        for line in f:
            n += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.write(json.dumps({k: r.get(k) for k in keep}, separators=(",", ":")) + "\n")
            w += 1
    print(f"  {slim.name}: read {n:,} wrote {w:,}")

full = DATA_DIR / "phase1_succteams_full.jsonl"
if not done(full):
    require(raw1, *RAW1)
    succ = set()
    with open(raw1, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if all(bool(objectives(r).get(k, False)) for k in OBJ):
                succ.add(r["team_id"])
    print(f"  teams with >=1 end-to-end success: {len(succ)}")
    keep = ("RowKey", "team_id", "scenario", "objectives", "started_time", "subject", "body")
    w = 0
    with open(raw1, encoding="utf-8") as f, open(full, "w", encoding="utf-8") as out:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("team_id") in succ:
                out.write(json.dumps({k: r.get(k) for k in keep}, separators=(",", ":")) + "\n")
                w += 1
    print(f"  {full.name}: wrote {w:,}")

# ---------------------------------------------------------------- Tensor Trust
print("Tensor Trust")
raw_tt = TT_DIR / "raw_dump_attacks.jsonl.bz2"

tt_slim = TT_DIR / "tt_slim.jsonl"
if not done(tt_slim):
    require(raw_tt, "Tensor Trust v2 raw attack dump",
            "Download raw-data/v2/raw_dump_attacks.jsonl.bz2 from "
            "github.com/HumanCompatibleAI/tensor-trust-data.")
    keep = ("attacker_id_anonymized", "defense_id", "timestamp", "attacker_input",
            "output_is_access_granted", "llm_choice", "defender_id_anonymized")
    n = w = 0
    with bz2.open(raw_tt, "rt", encoding="utf-8") as f, open(tt_slim, "w", encoding="utf-8") as out:
        for line in f:
            n += 1
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # self-attacks are practice against one's own defense; rows without a
            # defense_id cannot be assigned to a defense
            if str(r.get("is_self_attack")) == "True":
                continue
            if r.get("defense_id") in (None, "None", ""):
                continue
            out.write(json.dumps({k: r.get(k) for k in keep}, separators=(",", ":")) + "\n")
            w += 1
    print(f"  {tt_slim.name}: read {n:,} wrote {w:,}")

# ---------------------------------------------------------------- judge payloads
print("Section 7 judge payload set")
if (DATA_DIR / "judge_payloads.jsonl").exists() and not FORCE:
    print("  exists, skipping: judge_payloads.jsonl")
else:
    import subprocess
    r = subprocess.run([sys.executable, str(Path(__file__).parent / "14_build_judge_dataset.py")],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print("  " + (r.stdout.strip().splitlines()[-1] if r.returncode == 0 else "FAILED\n" + r.stderr[-800:]))

print("\npreprocessing complete")

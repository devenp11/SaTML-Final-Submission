"""Tensor Trust v2 loader. Cell = (attacker_id, defense_id): one attacker vs one fixed defense."""
import json, re, sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import TT_DIR  # env-overridable; see config.py


def tokens(s):
    return set(re.findall(r"\w+", (s or "").lower()))


def load_cells(min_attempts=2):
    """-> {(attacker, defense_id): [(ts, success, text, llm)]} sorted chronologically."""
    cells = defaultdict(list)
    with open(TT_DIR / "tt_slim.jsonl", encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            cells[(r["attacker_id_anonymized"], r["defense_id"])].append(
                (r.get("timestamp") or "",
                 str(r.get("output_is_access_granted")) == "True",
                 r.get("attacker_input") or "",
                 r.get("llm_choice")))
    out = {}
    for k, v in cells.items():
        if len(v) >= min_attempts:
            v.sort(key=lambda x: x[0])
            out[k] = v
    return out

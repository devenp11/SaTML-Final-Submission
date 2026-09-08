"""Loader for the LLMail-Inject challenge raw submissions."""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DATA_DIR, PHASE1_LEVELS, PHASE2_LEVELS

SCEN_RE = re.compile(r"^level(\d+)([a-z])$")
OBJ_KEYS = ["email.retrieved", "defense.undetected",
            "exfil.sent", "exfil.destination", "exfil.content"]


def parse_scenario(s):
    """'level2v' -> (2, 'v'). Returns (None, None) if unparseable."""
    m = SCEN_RE.match(s or "")
    return (int(m.group(1)), m.group(2)) if m else (None, None)


def iter_submissions(phase=2):
    """Yield normalised submission dicts from the raw JSONL."""
    levels = PHASE2_LEVELS if phase == 2 else PHASE1_LEVELS
    path = DATA_DIR / f"raw_submissions_phase{phase}.jsonl"
    if not path.exists():
        path = DATA_DIR / f"phase{phase}_slim.jsonl"   # streamed field subset
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            lvl, variant = parse_scenario(r.get("scenario"))
            if variant not in levels:
                continue
            model, defense = levels[variant]
            obj = r.get("objectives")
            if isinstance(obj, str):
                try:
                    obj = json.loads(obj)
                except json.JSONDecodeError:
                    obj = {}
            obj = obj or {}
            yield {
                "row_key": r.get("RowKey"),
                "team_id": r.get("team_id"),
                "level": lvl,
                "variant": variant,
                "model": model,
                "defense": defense,
                "started_time": r.get("started_time"),
                # objective flags
                **{k: bool(obj.get(k, False)) for k in OBJ_KEYS},
                # derived
                "evaded_defense": bool(obj.get("defense.undetected", False)),
                "end_to_end": all(bool(obj.get(k, False)) for k in OBJ_KEYS),
            }

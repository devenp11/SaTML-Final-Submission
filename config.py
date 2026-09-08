"""Paths and dataset locations.

No absolute or machine-specific paths. Data locations default to ./data inside the
repository and can be redirected with environment variables, so a reviewer can place
the (large) corpora anywhere:

    LLMAIL_DIR       default ./data/llmail-inject
    TENSORTRUST_DIR  default ./data/tensortrust

See README.md for how to obtain both public datasets.
"""
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parent
RESULTS_DIR = REPO_DIR / "results"
# Raw per-payload judge outputs backing Section 7. Shipped with the artifact, so the
# Section 7 analysis reproduces without a GPU. See results/judge_raw/MANIFEST.json.
JUDGE_RAW_DIR = RESULTS_DIR / "judge_raw"
FIGURES_DIR = REPO_DIR / "figures"

DATA_DIR = Path(os.environ.get("LLMAIL_DIR", REPO_DIR / "data" / "llmail-inject"))
TT_DIR = Path(os.environ.get("TENSORTRUST_DIR", REPO_DIR / "data" / "tensortrust"))

# LLMail-Inject level codes -> (target model, defense). The challenge design is fully
# crossed, so these let every analysis hold the target model fixed.
PHASE1_LEVELS = {
    "a": ("Phi-3", "PromptShield-v1"), "b": ("GPT", "PromptShield-v1"),
    "c": ("Phi-3", "TaskTracker-Phi3"), "d": ("GPT", "TaskTracker-Phi3"),
    "e": ("Phi-3", "Spotlight"),       "f": ("GPT", "Spotlight"),
    "g": ("Phi-3", "LLMJudge"),        "h": ("GPT", "LLMJudge"),
    "i": ("Phi-3", "AllCombined"),     "j": ("GPT", "AllCombined"),
}

PHASE2_LEVELS = {
    "k": ("Phi-3", "PromptShield-v2"), "l": ("GPT", "PromptShield-v2"),
    "m": ("Phi-3", "TaskTracker-Phi3"), "n": ("GPT", "TaskTracker-Phi3"),
    "o": ("Phi-3", "TaskTracker-Phi3.5MoE"), "p": ("GPT", "TaskTracker-Phi3.5MoE"),
    "q": ("Phi-3", "TaskTracker-Phi4"), "r": ("GPT", "TaskTracker-Phi4"),
    "s": ("Phi-3", "LLMJudge"),        "t": ("GPT", "LLMJudge"),
    "u": ("Phi-3", "AllCombined"),     "v": ("GPT", "AllCombined"),
}


def require(path: Path, what: str, how: str) -> Path:
    """Fail with an actionable message rather than a bare FileNotFoundError."""
    if not path.exists():
        raise SystemExit(
            f"\nMissing {what}: {path}\n"
            f"{how}\n"
            f"Set LLMAIL_DIR / TENSORTRUST_DIR to point elsewhere. See README.md.\n")
    return path

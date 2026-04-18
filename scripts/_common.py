"""Shared helpers for scripts/*.py — avoid importing from the root modules
which run their own main() on import."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
API_DIR = ROOT / "data"
PATTERNS_DIR = ROOT / "patterns"
ANALYSIS_DIR = ROOT / "analysis"
# main pipeline scripts live in src/ — add to path for imports
sys.path.insert(0, str(ROOT / "src"))


def load_records():
    """Walk data/<model>/<condition>.jsonl and yield all records."""
    out = []
    if not API_DIR.exists():
        return out
    for model_dir in sorted(API_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        for jsonl in sorted(model_dir.glob("*.jsonl")):
            with jsonl.open(encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    if r.get("reply"):
                        out.append(r)
    return out


def load_patterns(files=None):
    """Return {name: (compiled_re, meta)} from patterns/*.yaml.
    If files is given, only load those (basename match)."""
    import re
    import yaml
    compiled = {}
    for path in sorted(PATTERNS_DIR.glob("*.yaml")):
        if files is not None and path.name not in files:
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        for p in data.get("patterns", []):
            flags = re.MULTILINE if p.get("multiline") else 0
            compiled[p["name"]] = (
                re.compile(p["regex"], flags),
                {
                    "desc": p.get("desc", ""),
                    "source": p.get("source", ""),
                    "regex": p["regex"],
                    "file": path.name,
                },
            )
    return compiled


MODELS_MAIN = [
    "gpt-4o-2024-11-20",
    "gpt-5-chat-latest",
    "gpt-5.3-chat",
    "gpt-5.4",
    "claude-opus-4-6",
    "claude-opus-4-7",
]
CONDITIONS = ["A_empty", "B_listener", "C_poster"]

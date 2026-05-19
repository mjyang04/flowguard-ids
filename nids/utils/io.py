from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` and its parents if needed, then return it as a ``Path``."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Any, path: str | Path) -> None:
    """Write JSON data with UTF-8 encoding and stable indentation."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON file and return the decoded object."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_pickle(data: Any, path: str | Path) -> None:
    """Persist Python data with joblib, creating the parent directory first."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(data, p)


def load_pickle(path: str | Path) -> Any:
    """Load a joblib-serialised object from disk."""
    return joblib.load(path)

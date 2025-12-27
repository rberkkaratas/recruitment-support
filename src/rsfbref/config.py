from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class Config:
    raw: dict
    root: Path

def load_config(path: str | Path = "configs/v1.yaml") -> Config:
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    return Config(raw=raw, root=Path.cwd())

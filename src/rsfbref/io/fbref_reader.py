from __future__ import annotations
from pathlib import Path
import soccerdata as sd

def make_fbref(leagues, seasons, data_dir: str, no_cache: bool, no_store: bool):
    # soccerdata caches downloads under data_dir; keep this project-local. :contentReference[oaicite:7]{index=7}
    return sd.FBref(
        leagues=leagues,
        seasons=seasons,
        data_dir=Path(data_dir),
        no_cache=no_cache,
        no_store=no_store,
    )

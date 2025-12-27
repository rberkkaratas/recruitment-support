from __future__ import annotations
import hashlib
import pandas as pd

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def add_ids(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    def mk_player_id(r) -> str:
        parts = [
            str(r.get("player", "")).strip().lower(),
            str(r.get("nation", "")).strip().lower(),
            str(r.get("born", "")).strip().lower(),
        ]
        return _sha1("|".join(parts))

    def mk_team_id(r) -> str:
        parts = [
            str(r.get("team", "")).strip().lower(),
            str(r.get("league", "")).strip().lower(),
            str(r.get("season", "")).strip().lower(),
        ]
        return _sha1("|".join(parts))

    out["player_id"] = out.apply(mk_player_id, axis=1)
    out["team_id"] = out.apply(mk_team_id, axis=1)
    return out

def build_dim_player(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "player_id", "player", "nation", "age", "born", "pos", "position_bucket"
    ]
    out = df[cols].drop_duplicates(subset=["player_id"]).copy()
    out = out.rename(columns={"player": "player_name", "pos": "position_raw"})
    return out

def build_dim_team(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["team_id", "team", "league", "season"]
    out = df[cols].drop_duplicates(subset=["team_id"]).copy()
    out = out.rename(columns={"team": "team_name"})
    return out

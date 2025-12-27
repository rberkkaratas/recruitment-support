from __future__ import annotations

import pandas as pd
from .flatten import flatten_columns

PLAYER_STAT_TYPES_V1 = [
    "standard",
    "passing",
    "defense",
    "possession",
    "goal_shot_creation",
    "shooting",
    "playing_time",
    "misc",
]

KEY = ["league", "season", "team", "player"]

# Columns repeated across FBref tables; keep only once (from standard)
ENTITY_COLS = {"nation", "pos", "age", "born", "90s"}


def read_player_season_bundle(fbref) -> dict[str, pd.DataFrame]:
    bundle: dict[str, pd.DataFrame] = {}
    for st in PLAYER_STAT_TYPES_V1:
        df = fbref.read_player_season_stats(stat_type=st)
        df = df.reset_index()
        df = flatten_columns(df)
        bundle[st] = df
    return bundle


def _prefix_non_key_cols(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        if c in KEY:
            continue
        rename[c] = f"{prefix}__{c}"
    return df.rename(columns=rename)


def build_player_season_base(bundle: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all player-season stat tables on (league, season, team, player).
    Keep entity columns only once, and prefix other stat columns by stat_type.
    """
    base = bundle["standard"].copy()

    for st, df in bundle.items():
        if st == "standard":
            continue

        df2 = df.copy()

        # Drop overlapping entity columns from non-standard tables
        drop_cols = [c for c in df2.columns if c in ENTITY_COLS]
        df2 = df2.drop(columns=drop_cols, errors="ignore")

        # Prefix remaining non-key columns
        df2 = _prefix_non_key_cols(df2, st)

        # Merge only prefixed cols (plus keys)
        merge_cols = KEY + [c for c in df2.columns if c.startswith(f"{st}__")]
        base = base.merge(df2[merge_cols], on=KEY, how="left", validate="1:1")

    return base

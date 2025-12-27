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

# v2: make merge key more unique than (league, season, team, player)
# nation + born are present across FBref player season tables and disambiguate same-name players.
KEY = ["league", "season", "team", "player", "nation", "born"]

# Columns repeated across FBref tables; we still prefer taking them from "standard"
# but we must NOT drop those that are part of KEY, otherwise we can't merge.
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


def _dedupe_on_key(df: pd.DataFrame, st: str) -> pd.DataFrame:
    # Defensive: FBref tables can include duplicate rows for the same KEY.
    # Keep first; if you want to be stricter later, we can add diagnostics.
    before = len(df)
    df2 = df.drop_duplicates(subset=KEY, keep="first").copy()
    after = len(df2)
    if after < before:
        print(f"[dedupe] {st}: dropped {before - after} duplicate rows on KEY={KEY}")
    return df2


def build_player_season_base(bundle: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all player-season stat tables on v2 KEY.
    Keep entity columns only once (from standard), and prefix other stat columns by stat_type.
    """
    base = bundle["standard"].copy()

    # Ensure base is unique on KEY
    base = _dedupe_on_key(base, "standard")

    for st, df in bundle.items():
        if st == "standard":
            continue

        df2 = df.copy()

        # Dedupe on KEY before any merging
        df2 = _dedupe_on_key(df2, st)

        # Drop overlapping entity columns from non-standard tables,
        # but DO NOT drop any columns that are part of KEY.
        drop_cols = [c for c in df2.columns if (c in ENTITY_COLS and c not in KEY)]
        df2 = df2.drop(columns=drop_cols, errors="ignore")

        # Prefix remaining non-key columns
        df2 = _prefix_non_key_cols(df2, st)

        # Merge only prefixed cols (plus keys)
        merge_cols = KEY + [c for c in df2.columns if c.startswith(f"{st}__")]
        base = base.merge(df2[merge_cols], on=KEY, how="left", validate="1:1")

    return base

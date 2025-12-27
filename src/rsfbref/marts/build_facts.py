from __future__ import annotations
import pandas as pd


def _unique_preserve_order(items: list[str]) -> list[str]:
    """
    Return a list with duplicates removed, preserving order of first occurrence.
    Prevents pandas/pyarrow failures caused by duplicated column selections.
    """
    seen: set[str] = set()
    out: list[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _dedupe_columns(df: pd.DataFrame, context: str = "") -> pd.DataFrame:
    """
    Drop duplicated column names (keep first). This is a defensive safeguard:
    parquet writers and some pandas operations error on duplicate columns.
    """
    if df.columns.duplicated().any():
        dupes = df.columns[df.columns.duplicated()].tolist()
        msg = f"[dedupe_columns]{' ' + context if context else ''}: dropped duplicated columns: {dupes}"
        print(msg)
        df = df.loc[:, ~df.columns.duplicated()].copy()
    return df


KEY_FACT = ["player_team_season_id"]


def build_fact_player_season(df: pd.DataFrame) -> pd.DataFrame:
    """
    v2 fact table at strict grain: one row per player_team_season_id.

    Includes:
      - identifiers
      - minutes + position bucket
      - canonical engineered metrics
      - default-scope percentiles (pct_*)
      - role scores (score_*)
    """
    df = _dedupe_columns(df, context="input:build_fact_player_season")

    keep_prefixes = ("pct_", "score_")

    base_cols = [
        "player_team_season_id",
        "player_id", "team_id", "league", "season",
        "minutes", "nineties", "position_bucket",
        "pct_scope_default",
        # canonical metrics
        "pass_cmp_pct", "passes_att_p90", "prog_passes_p90", "passes_final_third_p90",
        "long_pass_cmp_pct", "key_passes_p90", "xa_p90", "crosses_pa_p90",
        "tkl_int_p90", "clr_p90", "errors_p90", "aerial_win_pct",
        "prog_carries_p90", "carries_pa_p90", "succ_takeons_p90", "takeon_succ_pct",
        "sca_p90", "mis_dis_p90", "fouls_p90", "Per_90_Minutes_npxG",
    ]

    # keep any existing pct_*, score_* columns (default scoring lens)
    extra = [c for c in df.columns if isinstance(c, str) and c.startswith(keep_prefixes)]

    cols = [c for c in (base_cols + extra) if c in df.columns]
    cols = _unique_preserve_order(cols)

    out = df[cols].copy()
    out = _dedupe_columns(out, context="output:build_fact_player_season")

    # enforce uniqueness at strict grain
    if "player_team_season_id" in out.columns:
        out = out.drop_duplicates(subset=KEY_FACT, keep="first").copy()

    return out


def build_fact_role_profile_card_v2(
    scored_df: pd.DataFrame,
    percentiles_long: pd.DataFrame,
) -> pd.DataFrame:
    """
    v2 interpretable layer (long):
      player_team_season_id x role_id x kpi_name x pct_scope

    percentiles_long schema expected:
      player_team_season_id, player_id, team_id, league, season, position_bucket, minutes,
      kpi_name, kpi_value, kpi_pct, pct_scope
    """
    scored_df = _dedupe_columns(scored_df, context="input:build_fact_role_profile_card_v2:scored")
    percentiles_long = _dedupe_columns(percentiles_long, context="input:build_fact_role_profile_card_v2:pct_long")

    roles = [c for c in scored_df.columns if isinstance(c, str) and c.startswith("score_")]
    role_ids = [c.replace("score_", "") for c in roles]

    id_cols = [
        "player_team_season_id", "player_id", "team_id", "league", "season",
        "minutes", "position_bucket"
    ]
    id_cols = [c for c in id_cols if c in scored_df.columns]

    # Build role header (one row per eligible player_team_season_id x role)
    headers = []
    for role_id in role_ids:
        score_col = f"score_{role_id}"
        if score_col not in scored_df.columns:
            continue
        tmp = scored_df[id_cols].copy()
        tmp["role_id"] = role_id
        tmp["role_score"] = scored_df[score_col]
        headers.append(tmp)

    if not headers:
        return pd.DataFrame()

    header = pd.concat(headers, ignore_index=True)
    header = header[header["role_score"].notna()].copy()

    # Join to long percentiles (1:m)
    join_keys = ["player_team_season_id", "player_id", "team_id", "league", "season", "position_bucket", "minutes"]
    join_keys = [k for k in join_keys if k in header.columns and k in percentiles_long.columns]

    out = header.merge(
        percentiles_long,
        on=join_keys,
        how="left",
        validate="1:m",
    )

    out = _dedupe_columns(out, context="output:build_fact_role_profile_card_v2")
    return out

from __future__ import annotations
import pandas as pd

KEY_FACT = ["player_id", "team_id", "league", "season"]

def build_fact_player_season(df: pd.DataFrame) -> pd.DataFrame:
    # Keep identity + minutes + canonical metrics + percentiles + role scores
    keep_prefixes = ("pct_", "score_")
    keep_cols = [
        "player_id", "team_id", "league", "season",
        "minutes", "nineties", "position_bucket",
        # canonical metrics (from clean)
        "pass_cmp_pct", "passes_att_p90", "prog_passes_p90", "passes_final_third_p90",
        "long_pass_cmp_pct", "key_passes_p90", "xa_p90", "crosses_pa_p90",
        "tkl_int_p90", "clr_p90", "errors_p90", "aerial_win_pct",
        "prog_carries_p90", "carries_pa_p90", "succ_takeons_p90", "takeon_succ_pct",
        "sca_p90", "mis_dis_p90", "fouls_p90", "Per_90_Minutes_npxG",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    extra = [c for c in df.columns if c.startswith(keep_prefixes)]
    out = df[keep_cols + extra].copy()

    # enforce uniqueness
    out = out.drop_duplicates(subset=KEY_FACT)
    return out

def build_fact_role_profile_card(df: pd.DataFrame) -> pd.DataFrame:
    """
    Long-format KPI rows per (player_id, team_id, season, role).
    Tableau loves this for radar-like views / bars / tooltips.
    """
    roles = [c for c in df.columns if c.startswith("score_")]
    role_ids = [c.replace("score_", "") for c in roles]

    # define KPI set for cards (v1)
    kpis = [
        "pass_cmp_pct", "passes_att_p90", "prog_passes_p90", "passes_final_third_p90",
        "long_pass_cmp_pct", "key_passes_p90", "xa_p90", "crosses_pa_p90",
        "tkl_int_p90", "clr_p90", "errors_p90", "aerial_win_pct",
        "prog_carries_p90", "carries_pa_p90", "succ_takeons_p90", "takeon_succ_pct",
        "sca_p90", "mis_dis_p90", "fouls_p90", "Per_90_Minutes_npxG",
    ]
    kpis = [k for k in kpis if k in df.columns]

    rows = []
    for role_id in role_ids:
        score_col = f"score_{role_id}"
        tmp = df[["player_id", "team_id", "league", "season", "minutes", "position_bucket"]].copy()
        tmp["role_id"] = role_id
        tmp["role_score"] = df[score_col]

        # attach KPI + percentile as long table
        for k in kpis:
            out = tmp.copy()
            out["kpi_name"] = k
            out["kpi_value"] = df[k]
            pct_col = f"pct_{k}"
            out["kpi_pct"] = df[pct_col] if pct_col in df.columns else pd.NA
            rows.append(out)

    card = pd.concat(rows, ignore_index=True)
    # Keep only players who have a role score (eligible)
    card = card[card["role_score"].notna()].copy()
    return card

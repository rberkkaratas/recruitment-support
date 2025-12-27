from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.metrics.pairwise import cosine_distances

# Percentile-feature sets per role (input df must contain these columns).
ROLE_FEATURES: dict[str, list[str]] = {
    "BPCB": [
        "pct_prog_passes_p90",
        "pct_passes_final_third_p90",
        "pct_pass_cmp_pct",
        "pct_long_pass_cmp_pct",
        "pct_tkl_int_p90",
        "pct_clr_p90",
        "pct_aerial_win_pct",
        "pct_prog_carries_p90",
        "pct_errors_p90",
    ],
    "DLP": [
        "pct_passes_att_p90",
        "pct_pass_cmp_pct",
        "pct_prog_passes_p90",
        "pct_passes_final_third_p90",
        "pct_key_passes_p90",
        "pct_xa_p90",
        "pct_sca_p90",
        "pct_prog_carries_p90",
        "pct_tkl_int_p90",
        "pct_fouls_p90",
    ],
    "WCR": [
        "pct_prog_carries_p90",
        "pct_carries_pa_p90",
        "pct_succ_takeons_p90",
        "pct_takeon_succ_pct",
        "pct_xa_p90",
        "pct_key_passes_p90",
        "pct_sca_p90",
        "pct_crosses_pa_p90",
        "pct_Per_90_Minutes_npxG",
        "pct_mis_dis_p90",
    ],
}

# Which percentile features should be inverted (higher is worse).
INVERT_PCT_FEATURES: set[str] = {
    "pct_errors_p90",
    "pct_fouls_p90",
    "pct_mis_dis_p90",
}


def _top_reason_codes(
    anchor_vec: np.ndarray,
    comp_vec: np.ndarray,
    feat_names: list[str],
    k: int = 3,
) -> list[str]:
    """
    Return human-readable "why similar/different" indicators.
    Uses absolute standardized deltas to pick top contributors.
    """
    diffs = comp_vec - anchor_vec
    idx = np.argsort(np.abs(diffs))[::-1][:k]

    reasons: list[str] = []
    for i in idx:
        sign = "higher" if diffs[i] > 0 else "lower"
        fname = feat_names[i].replace("pct_", "")
        reasons.append(f"{fname}:{sign}")
    return reasons


def build_fact_comparables(
    df: pd.DataFrame,
    role_id: str,
    top_n: int = 10,
    comparison_scope: str = "league_season",
    pct_scope: str = "league_season",
) -> pd.DataFrame:
    """
    Build comparables within the eligible pool for a given role.

    Requirements:
      - df contains: player_team_season_id, player_id, team_id, league, season
      - df contains: score_{role_id} (notna marks eligibility)
      - df contains percentile columns used in ROLE_FEATURES[role_id]
        (pct_scope indicates which lens they represent; script ensures they exist)

    Output schema (Tableau-friendly):
      comparison_scope, pct_scope, role_id,
      anchor_pts_id, anchor_player_id, anchor_team_id, anchor_league, anchor_season,
      comp_pts_id, comp_player_id, comp_team_id, comp_league, comp_season,
      different_league, different_season,
      distance, rank, reason_1..reason_3
    """
    score_col = f"score_{role_id}"
    required = {"player_team_season_id", "player_id", "team_id", "league", "season"}

    if score_col not in df.columns or not required.issubset(df.columns):
        return pd.DataFrame(columns=[
            "comparison_scope", "pct_scope", "role_id",
            "anchor_pts_id", "anchor_player_id", "anchor_team_id", "anchor_league", "anchor_season",
            "comp_pts_id", "comp_player_id", "comp_team_id", "comp_league", "comp_season",
            "different_league", "different_season",
            "distance", "rank", "reason_1", "reason_2", "reason_3",
        ])

    feats = [f for f in ROLE_FEATURES.get(role_id, []) if f in df.columns]
    if not feats:
        return pd.DataFrame(columns=[
            "comparison_scope", "pct_scope", "role_id",
            "anchor_pts_id", "anchor_player_id", "anchor_team_id", "anchor_league", "anchor_season",
            "comp_pts_id", "comp_player_id", "comp_team_id", "comp_league", "comp_season",
            "different_league", "different_season",
            "distance", "rank", "reason_1", "reason_2", "reason_3",
        ])

    pool = df[df[score_col].notna()].copy()
    if len(pool) < 2:
        return pd.DataFrame(columns=[
            "comparison_scope", "pct_scope", "role_id",
            "anchor_pts_id", "anchor_player_id", "anchor_team_id", "anchor_league", "anchor_season",
            "comp_pts_id", "comp_player_id", "comp_team_id", "comp_league", "comp_season",
            "different_league", "different_season",
            "distance", "rank", "reason_1", "reason_2", "reason_3",
        ])

    eff_top_n = min(int(top_n), len(pool) - 1)

    # Build feature matrix (percentiles 0..100). Fill NaNs conservatively with median.
    X = pool[feats].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))

    # Invert "bad is high" features so similarity space aligns with role quality direction.
    for f in feats:
        if f in INVERT_PCT_FEATURES:
            X[f] = 100.0 - X[f]

    # Robust scale then compute cosine distances
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    D = cosine_distances(Xs, Xs)

    rows: list[dict] = []
    for i in range(len(pool)):
        anchor = pool.iloc[i]
        order = np.argsort(D[i])
        order = order[order != i][:eff_top_n]

        anchor_vec = Xs[i]
        for rank, j in enumerate(order, start=1):
            comp = pool.iloc[j]
            reasons = _top_reason_codes(anchor_vec, Xs[j], feats, k=3)

            rows.append({
                "comparison_scope": comparison_scope,
                "pct_scope": pct_scope,
                "role_id": role_id,

                "anchor_pts_id": anchor["player_team_season_id"],
                "anchor_player_id": anchor["player_id"],
                "anchor_team_id": anchor["team_id"],
                "anchor_league": anchor["league"],
                "anchor_season": anchor["season"],

                "comp_pts_id": comp["player_team_season_id"],
                "comp_player_id": comp["player_id"],
                "comp_team_id": comp["team_id"],
                "comp_league": comp["league"],
                "comp_season": comp["season"],

                "different_league": bool(anchor["league"] != comp["league"]),
                "different_season": bool(anchor["season"] != comp["season"]),

                "distance": float(D[i, j]),
                "rank": int(rank),
                "reason_1": reasons[0] if len(reasons) > 0 else None,
                "reason_2": reasons[1] if len(reasons) > 1 else None,
                "reason_3": reasons[2] if len(reasons) > 2 else None,
            })

    return pd.DataFrame(rows)

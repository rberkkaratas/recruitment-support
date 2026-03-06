"""Tests for rsfbref.analytics.roles - covers bug fixes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from rsfbref.analytics.roles import apply_must_haves, load_roles, score_roles


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROLES_YAML_CONTENT = {
    "roles": [
        {
            "role_id": "BPCB",
            "role_name": "Ball-Playing CB",
            "position_bucket": "CB",
            "must_have": {"min_minutes": 630},
            "weights": {
                "prog_passes_p90": 0.16,
                "tackles_interceptions_p90": 0.14,  # maps to tkl_int_p90
                "clearances_p90": 0.06,              # maps to clr_p90
                "aerial_win_pct_or_won_p90": 0.12,  # maps to aerial_win_pct
                "errors_or_dispossessed_neg": 0.10, # maps to errors_p90
                "pass_cmp_pct": 0.42,
            },
            "negative_metrics": ["errors_or_dispossessed_neg"],
        },
        {
            "role_id": "DLP",
            "role_name": "Deep-Lying Playmaker",
            "position_bucket": "DMCM",
            "must_have": {"min_minutes": 900},
            "weights": {
                "passes_att_p90": 0.40,
                "tackles_interceptions_p90": 0.40,  # maps to tkl_int_p90
                "fouls_committed_neg": 0.20,        # maps to fouls_p90
            },
            "negative_metrics": ["fouls_committed_neg"],
        },
        {
            "role_id": "WCR",
            "role_name": "Winger Creator",
            "position_bucket": "WIDE",
            "must_have": {"min_minutes": 900},
            "weights": {
                "xa_p90": 0.50,
                "npxg_p90": 0.50,  # maps to Per_90_Minutes_npxG
            },
            "negative_metrics": [],
        },
    ]
}


def _write_roles_yaml(content: dict, tmp_path: Path) -> str:
    path = tmp_path / "roles.yaml"
    path.write_text(yaml.dump(content), encoding="utf-8")
    return str(path)


def _make_df(**kwargs) -> pd.DataFrame:
    """Build a minimal single-row player DataFrame."""
    defaults = {
        "position_bucket": "CB",
        "minutes": 1000,
        # canonical percentile columns
        "pct_prog_passes_p90": 70.0,
        "pct_tkl_int_p90": 80.0,
        "pct_clr_p90": 60.0,
        "pct_aerial_win_pct": 75.0,
        "pct_errors_p90": 30.0,
        "pct_pass_cmp_pct": 65.0,
        "pct_passes_att_p90": 55.0,
        "pct_fouls_p90": 40.0,
        "pct_xa_p90": 85.0,
        "pct_Per_90_Minutes_npxG": 90.0,
    }
    defaults.update(kwargs)
    return pd.DataFrame([defaults])


# ---------------------------------------------------------------------------
# load_roles
# ---------------------------------------------------------------------------


def test_load_roles_returns_list(tmp_path):
    path = _write_roles_yaml(ROLES_YAML_CONTENT, tmp_path)
    roles = load_roles(path)
    assert isinstance(roles, list)
    assert len(roles) == 3
    assert roles[0]["role_id"] == "BPCB"


def test_load_roles_no_file_handle_leak(tmp_path):
    """load_roles must not leave an open file handle (uses Path.read_text)."""
    path = _write_roles_yaml(ROLES_YAML_CONTENT, tmp_path)
    # Call repeatedly to ensure no resource leak / OSError from too many open files
    for _ in range(50):
        load_roles(path)
    # No assertion needed: the test passes if no exception is raised


# ---------------------------------------------------------------------------
# score_roles – mapping bugs
# ---------------------------------------------------------------------------


def test_bpcb_tackles_interceptions_mapped(tmp_path):
    """tackles_interceptions_p90 must map to pct_tkl_int_p90 (not skipped)."""
    path = _write_roles_yaml(ROLES_YAML_CONTENT, tmp_path)

    # Player A: high tkl_int percentile
    df_high = _make_df(pct_tkl_int_p90=99.0)
    # Player B: low tkl_int percentile
    df_low = _make_df(pct_tkl_int_p90=1.0)

    scored_high = score_roles(df_high, roles_yaml_path=path)["score_BPCB"].iloc[0]
    scored_low = score_roles(df_low, roles_yaml_path=path)["score_BPCB"].iloc[0]

    # The tackle/interception feature has a non-trivial weight (0.14), so the
    # scores should differ meaningfully between high and low values.
    assert pd.notna(scored_high), "BPCB score should not be NA for eligible player"
    assert pd.notna(scored_low), "BPCB score should not be NA for eligible player"
    assert scored_high > scored_low, (
        "Player with higher pct_tkl_int_p90 should score higher on BPCB"
    )


def test_bpcb_clearances_mapped(tmp_path):
    """clearances_p90 must map to pct_clr_p90 (not skipped)."""
    path = _write_roles_yaml(ROLES_YAML_CONTENT, tmp_path)

    df_high = _make_df(pct_clr_p90=99.0)
    df_low = _make_df(pct_clr_p90=1.0)

    scored_high = score_roles(df_high, roles_yaml_path=path)["score_BPCB"].iloc[0]
    scored_low = score_roles(df_low, roles_yaml_path=path)["score_BPCB"].iloc[0]

    assert scored_high > scored_low, (
        "Player with higher pct_clr_p90 should score higher on BPCB"
    )


def test_dlp_tackles_interceptions_mapped(tmp_path):
    """tackles_interceptions_p90 must map to pct_tkl_int_p90 for DLP."""
    path = _write_roles_yaml(ROLES_YAML_CONTENT, tmp_path)

    df_high = _make_df(position_bucket="DMCM", pct_tkl_int_p90=99.0)
    df_low = _make_df(position_bucket="DMCM", pct_tkl_int_p90=1.0)

    scored_high = score_roles(df_high, roles_yaml_path=path)["score_DLP"].iloc[0]
    scored_low = score_roles(df_low, roles_yaml_path=path)["score_DLP"].iloc[0]

    assert pd.notna(scored_high)
    assert pd.notna(scored_low)
    assert scored_high > scored_low, (
        "Player with higher pct_tkl_int_p90 should score higher on DLP"
    )


def test_wcr_npxg_mapped(tmp_path):
    """npxg_p90 must map to pct_Per_90_Minutes_npxG (not skipped)."""
    path = _write_roles_yaml(ROLES_YAML_CONTENT, tmp_path)

    df_high = _make_df(position_bucket="WIDE", pct_Per_90_Minutes_npxG=99.0)
    df_low = _make_df(position_bucket="WIDE", pct_Per_90_Minutes_npxG=1.0)

    scored_high = score_roles(df_high, roles_yaml_path=path)["score_WCR"].iloc[0]
    scored_low = score_roles(df_low, roles_yaml_path=path)["score_WCR"].iloc[0]

    assert pd.notna(scored_high), "WCR score should not be NA for eligible player"
    assert pd.notna(scored_low), "WCR score should not be NA for eligible player"
    assert scored_high > scored_low, (
        "Player with higher pct_Per_90_Minutes_npxG should score higher on WCR"
    )


def test_score_uses_full_weight_sum(tmp_path):
    """
    When all mapped features exist, wsum should equal the sum of all weights in
    the config (no silent skipping), so score is normalised correctly.
    """
    path = _write_roles_yaml(ROLES_YAML_CONTENT, tmp_path)

    # All features at 50th percentile → score should be ~50
    df = _make_df(
        pct_prog_passes_p90=50.0,
        pct_tkl_int_p90=50.0,
        pct_clr_p90=50.0,
        pct_aerial_win_pct=50.0,
        pct_errors_p90=50.0,  # inverted: 100-50=50
        pct_pass_cmp_pct=50.0,
    )
    scored = score_roles(df, roles_yaml_path=path)
    score = scored["score_BPCB"].iloc[0]
    assert abs(score - 50.0) < 1e-6, (
        f"With all features at p50, BPCB score should be 50, got {score}"
    )


def test_ineligible_player_gets_na(tmp_path):
    """Players outside the position bucket should receive NA score."""
    path = _write_roles_yaml(ROLES_YAML_CONTENT, tmp_path)
    df = _make_df(position_bucket="WIDE")  # not CB
    scored = score_roles(df, roles_yaml_path=path)
    assert pd.isna(scored["score_BPCB"].iloc[0])

from __future__ import annotations
from dataclasses import dataclass

# Scope name -> groupby columns for percentile computation
SCOPE_GROUPS: dict[str, list[str]] = {
    # safest: compare within league+season (+ bucket)
    "league_season": ["league", "season", "position_bucket"],
    # within league across multiple seasons
    "league_multi_season": ["league", "position_bucket"],
    # across leagues within the same season
    "multi_league_season": ["season", "position_bucket"],
    # global pool (still bucketed)
    "multi_league_multi_season": ["position_bucket"],
}

@dataclass(frozen=True)
class ScopeSpec:
    name: str
    group_cols: list[str]

def get_scope_spec(scope_name: str) -> ScopeSpec:
    if scope_name not in SCOPE_GROUPS:
        raise ValueError(f"Unknown scope_name={scope_name}. Supported: {list(SCOPE_GROUPS)}")
    return ScopeSpec(name=scope_name, group_cols=SCOPE_GROUPS[scope_name])

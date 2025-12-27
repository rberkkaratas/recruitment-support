from __future__ import annotations
from pathlib import Path
import typer
import pandas as pd

from rsfbref.config import load_config
from rsfbref.io.fbref_reader import make_fbref
from rsfbref.transform.player_season import read_player_season_bundle, build_player_season_base
from rsfbref.transform.clean_player_season import build_player_season_clean
from rsfbref.marts.build_dims import add_ids
from rsfbref.features.scopes import get_scope_spec
from rsfbref.features.percentiles import add_percentiles_wide
from rsfbref.analytics.roles import score_roles

app = typer.Typer()

@app.command()
def main(config: str = "configs/v2.yaml"):
    cfg = load_config(config).raw

    fbref = make_fbref(
        leagues=cfg["fbref"]["leagues"],
        seasons=cfg["fbref"]["seasons"],
        data_dir=cfg["fbref"]["data_dir"],
        no_cache=cfg["fbref"]["no_cache"],
        no_store=cfg["fbref"]["no_store"],
    )

    bundle = read_player_season_bundle(fbref)
    base = build_player_season_base(bundle)

    Path("data/intermediate").mkdir(parents=True, exist_ok=True)
    base.to_parquet("data/intermediate/player_season_base.parquet", index=False)

    clean = build_player_season_clean(
        base,
        min_minutes=cfg["filters"]["min_minutes"],
    )
    clean.to_parquet("data/intermediate/player_season_clean.parquet", index=False)

    # ids + strict grain key
    clean = add_ids(clean)

    # metrics present for percentiles/scoring
    metric_cols = [
        "pass_cmp_pct", "passes_att_p90", "prog_passes_p90", "passes_final_third_p90",
        "long_pass_cmp_pct", "key_passes_p90", "xa_p90", "crosses_pa_p90",
        "tkl_int_p90", "clr_p90", "errors_p90", "aerial_win_pct",
        "prog_carries_p90", "carries_pa_p90", "succ_takeons_p90", "takeon_succ_pct",
        "sca_p90", "mis_dis_p90", "fouls_p90", "Per_90_Minutes_npxG"
    ]
    metric_cols = [c for c in metric_cols if c in clean.columns]

    # default percentile scope feeds role scoring + v2 "safe" lens
    default_scope = cfg["scopes"]["default_percentile_scope"]
    spec = get_scope_spec(default_scope)

    scored = add_percentiles_wide(
        clean,
        metric_cols=metric_cols,
        group_cols=spec.group_cols,
        prefix="pct_",
    )
    scored["pct_scope_default"] = default_scope

    # role scoring unchanged
    scored = score_roles(scored, roles_yaml_path=cfg["roles"]["role_defs_path"])

    scored.to_parquet("data/intermediate/player_season_scored.parquet", index=False)
    print(f"Wrote data/intermediate/player_season_scored.parquet ({len(scored):,} rows)")

if __name__ == "__main__":
    app()

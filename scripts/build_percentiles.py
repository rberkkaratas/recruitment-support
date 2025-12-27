from __future__ import annotations
from pathlib import Path
import typer
import pandas as pd

from rsfbref.config import load_config
from rsfbref.features.scopes import get_scope_spec
from rsfbref.features.percentiles import build_percentiles_long
from rsfbref.export.tableau import export_csv

app = typer.Typer()

@app.command()
def main(config: str = "configs/v2.yaml"):
    cfg = load_config(config).raw
    scopes: list[str] = cfg["scopes"]["percentile_scopes"]

    clean_path = Path("data/intermediate/player_season_clean.parquet")
    if not clean_path.exists():
        raise FileNotFoundError("Run scripts/run_pipeline.py first (it writes player_season_clean.parquet).")

    df = pd.read_parquet(clean_path)

    # v2 expects ids already present in clean? if not, read scored instead.
    scored_path = Path("data/intermediate/player_season_scored.parquet")
    if scored_path.exists():
        df = pd.read_parquet(scored_path)

    # id cols for long mart
    id_cols = ["player_team_season_id", "player_id", "team_id", "league", "season", "position_bucket", "minutes"]
    id_cols = [c for c in id_cols if c in df.columns]

    metric_cols = [
        "pass_cmp_pct", "passes_att_p90", "prog_passes_p90", "passes_final_third_p90",
        "long_pass_cmp_pct", "key_passes_p90", "xa_p90", "crosses_pa_p90",
        "tkl_int_p90", "clr_p90", "errors_p90", "aerial_win_pct",
        "prog_carries_p90", "carries_pa_p90", "succ_takeons_p90", "takeon_succ_pct",
        "sca_p90", "mis_dis_p90", "fouls_p90", "Per_90_Minutes_npxG"
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]

    parts = []
    for s in scopes:
        spec = get_scope_spec(s)
        long = build_percentiles_long(
            df=df,
            metric_cols=metric_cols,
            group_cols=spec.group_cols,
            pct_scope=s,
            id_cols=id_cols,
        )
        parts.append(long)

    out = pd.concat(parts, ignore_index=True)

    Path("data/marts").mkdir(parents=True, exist_ok=True)
    out.to_parquet("data/marts/fact_percentiles.parquet", index=False)

    out_csv = Path(cfg["exports"]["out_dir"]) / "fact_percentiles.csv"
    export_csv(out, out_csv)

    print(f"Wrote {len(out):,} rows -> {out_csv}")

if __name__ == "__main__":
    app()

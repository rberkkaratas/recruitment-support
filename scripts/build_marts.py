from __future__ import annotations
from pathlib import Path
import typer
import pandas as pd

from rsfbref.config import load_config
from rsfbref.marts.build_dims import build_dim_player, build_dim_team
from rsfbref.marts.build_facts import build_fact_player_season, build_fact_role_profile_card_v2
from rsfbref.export.tableau import export_csv, export_tableau_v1  # keep your existing exporter

app = typer.Typer()

@app.command()
def main(config: str = "configs/v2.yaml"):
    cfg = load_config(config).raw

    scored_path = Path("data/intermediate/player_season_scored.parquet")
    df = pd.read_parquet(scored_path)

    # dims
    dim_player = build_dim_player(df)
    dim_team = build_dim_team(df)

    # core fact (wide)
    fact_player_season = build_fact_player_season(df)

    # scope-aware profile card (long) requires fact_percentiles
    pct_path = Path("data/marts/fact_percentiles.parquet")
    if not pct_path.exists():
        raise FileNotFoundError("Run scripts/build_percentiles.py first (creates data/marts/fact_percentiles.parquet).")
    percentiles_long = pd.read_parquet(pct_path)

    fact_role_profile_card = build_fact_role_profile_card_v2(df, percentiles_long)

    Path("data/marts").mkdir(parents=True, exist_ok=True)
    dim_player.to_parquet("data/marts/dim_player.parquet", index=False)
    dim_team.to_parquet("data/marts/dim_team.parquet", index=False)
    fact_player_season.to_parquet("data/marts/fact_player_season.parquet", index=False)
    fact_role_profile_card.to_parquet("data/marts/fact_role_profile_card.parquet", index=False)

    # Tableau exports (core set) + percentiles/card
    out_dir = Path(cfg["exports"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    export_csv(dim_player, out_dir / "dim_player.csv")
    export_csv(dim_team, out_dir / "dim_team.csv")
    export_csv(fact_player_season, out_dir / "fact_player_season.csv")
    export_csv(fact_role_profile_card, out_dir / "fact_role_profile_card.csv")
    # fact_percentiles already exported by build_percentiles.py (safe to re-export too if you want)

    print("Wrote v2 marts parquet + Tableau CSV exports.")

if __name__ == "__main__":
    app()

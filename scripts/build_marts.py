from __future__ import annotations
from pathlib import Path
import typer
import pandas as pd

from rsfbref.config import load_config
from rsfbref.marts.build_dims import add_ids, build_dim_player, build_dim_team
from rsfbref.marts.build_facts import build_fact_player_season, build_fact_role_profile_card
from rsfbref.export.tableau import export_tableau_v1

app = typer.Typer()

@app.command()
def main(config: str = "configs/v1.yaml"):
    cfg = load_config(config).raw

    scored_path = Path("data/intermediate/player_season_scored.parquet")
    df = pd.read_parquet(scored_path)

    df = add_ids(df)

    dim_player = build_dim_player(df)
    dim_team = build_dim_team(df)
    fact_player_season = build_fact_player_season(df)
    fact_role_profile_card = build_fact_role_profile_card(df)

    Path("data/marts").mkdir(parents=True, exist_ok=True)
    dim_player.to_parquet("data/marts/dim_player.parquet", index=False)
    dim_team.to_parquet("data/marts/dim_team.parquet", index=False)
    fact_player_season.to_parquet("data/marts/fact_player_season.parquet", index=False)
    fact_role_profile_card.to_parquet("data/marts/fact_role_profile_card.parquet", index=False)

    export_tableau_v1(
        dim_player=dim_player,
        dim_team=dim_team,
        fact_player_season=fact_player_season,
        fact_role_profile_card=fact_role_profile_card,
        out_dir=cfg["exports"]["out_dir"],
    )

    print("Wrote marts parquet + Tableau CSV exports.")

if __name__ == "__main__":
    app()

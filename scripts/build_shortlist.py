from __future__ import annotations
from pathlib import Path
import typer
import pandas as pd

from rsfbref.analytics.shortlist import build_shortlist
from rsfbref.export.tableau import export_csv

app = typer.Typer()

@app.command()
def main(top_n: int = 50):
    fact = pd.read_parquet("data/marts/fact_player_season.parquet")
    dim_player = pd.read_parquet("data/marts/dim_player.parquet")[["player_id", "age"]]

    # bring age into shortlist input (left join, 1:1 on player_id)
    df = fact.merge(dim_player, on="player_id", how="left", validate="m:1")

    parts = []
    for role_id in ["BPCB", "DLP", "WCR"]:
        parts.append(build_shortlist(df, role_id=role_id, top_n=top_n))

    out = pd.concat(parts, ignore_index=True)

    Path("data/marts").mkdir(parents=True, exist_ok=True)
    out.to_parquet("data/marts/fact_shortlist.parquet", index=False)

    out_csv = Path("data/exports/tableau/fact_shortlist.csv")
    export_csv(out, out_csv)

    print(f"Wrote {len(out):,} rows -> {out_csv}")

if __name__ == "__main__":
    app()

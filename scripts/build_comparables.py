from __future__ import annotations
from pathlib import Path
import typer
import pandas as pd

from rsfbref.analytics.comparables import build_fact_comparables
from rsfbref.export.tableau import export_csv

app = typer.Typer()

@app.command()
def main(top_n: int = 10):
    df = pd.read_parquet("data/marts/fact_player_season.parquet")

    # safety: only same league+season already in this mart (v1), but keep as-is
    all_roles = ["BPCB", "DLP", "WCR"]
    parts = []
    for r in all_roles:
        parts.append(build_fact_comparables(df, role_id=r, top_n=top_n))

    out = pd.concat(parts, ignore_index=True)

    Path("data/marts").mkdir(parents=True, exist_ok=True)
    out.to_parquet("data/marts/fact_comparables.parquet", index=False)

    out_csv = Path("data/exports/tableau/fact_comparables.csv")
    export_csv(out, out_csv)

    print(f"Wrote {len(out):,} rows -> {out_csv}")

if __name__ == "__main__":
    app()

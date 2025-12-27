from __future__ import annotations
from pathlib import Path
import typer
import pandas as pd

from rsfbref.config import load_config
from rsfbref.analytics.comparables import build_fact_comparables
from rsfbref.export.tableau import export_csv

app = typer.Typer()

def _attach_pct_scope(df_fact: pd.DataFrame, pct_scope: str) -> pd.DataFrame:
    """
    Ensure df has pct_* columns for the chosen pct_scope.
    If pct_scope == df_fact['pct_scope_default'], we already have them.
    Otherwise, pull from fact_percentiles long and pivot.
    """
    default_scope = df_fact["pct_scope_default"].iloc[0] if "pct_scope_default" in df_fact.columns and len(df_fact) else None
    if pct_scope == default_scope:
        return df_fact

    pct = pd.read_parquet("data/marts/fact_percentiles.parquet")
    pct = pct[pct["pct_scope"] == pct_scope].copy()

    # pivot to wide pct_* columns
    wide = pct.pivot_table(
        index="player_team_season_id",
        columns="kpi_name",
        values="kpi_pct",
        aggfunc="first",
    ).reset_index()

    wide.columns = ["player_team_season_id"] + [f"pct_{c}" for c in wide.columns[1:]]
    out = df_fact.drop(columns=[c for c in df_fact.columns if c.startswith("pct_")], errors="ignore").merge(
        wide, on="player_team_season_id", how="left", validate="1:1"
    )
    return out

@app.command()
def main(config: str = "configs/v2.yaml", top_n: int = 10, pct_scope: str | None = None):
    cfg = load_config(config).raw

    df = pd.read_parquet("data/marts/fact_player_season.parquet")

    use_scope = pct_scope or cfg["scopes"]["comparison_scope"]
    df = _attach_pct_scope(df, pct_scope=use_scope)

    parts = []
    for r in ["BPCB", "DLP", "WCR"]:
        parts.append(build_fact_comparables(
            df,
            role_id=r,
            top_n=top_n,
            comparison_scope=cfg["scopes"]["comparison_scope"],
            pct_scope=use_scope,
        ))

    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    Path("data/marts").mkdir(parents=True, exist_ok=True)
    out.to_parquet("data/marts/fact_comparables.parquet", index=False)

    out_csv = Path(cfg["exports"]["out_dir"]) / "fact_comparables.csv"
    export_csv(out, out_csv)

    print(f"Wrote {len(out):,} rows -> {out_csv}")

if __name__ == "__main__":
    app()

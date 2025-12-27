from __future__ import annotations
from pathlib import Path
import pandas as pd

def export_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)

def export_tableau_v1(dim_player, dim_team, fact_player_season, fact_role_profile_card, out_dir: str):
    out = Path(out_dir)
    export_csv(dim_player, out / "dim_player.csv")
    export_csv(dim_team, out / "dim_team.csv")
    export_csv(fact_player_season, out / "fact_player_season.csv")
    export_csv(fact_role_profile_card, out / "fact_role_profile_card.csv")

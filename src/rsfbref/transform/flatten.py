from __future__ import annotations
import pandas as pd

def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    soccerdata often returns MultiIndex columns (e.g., passing breakdowns).
    This flattens to snake_case-ish strings with minimal surprises.
    """
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            "_".join([str(x).strip() for x in tup if x and str(x).strip() != "nan"]).strip("_")
            for tup in out.columns.to_list()
        ]
    out.columns = [c.strip().replace(" ", "_").replace("%", "pct") for c in out.columns]
    return out

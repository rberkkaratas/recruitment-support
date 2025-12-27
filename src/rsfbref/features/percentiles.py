from __future__ import annotations
import pandas as pd

def add_percentiles(df: pd.DataFrame, metric_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in metric_cols:
        out[f"pct_{col}"] = out.groupby(group_cols, dropna=False)[col].transform(
            lambda s: s.rank(pct=True, method="average") * 100
        )
    return out

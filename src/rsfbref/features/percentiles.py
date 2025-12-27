from __future__ import annotations
import pandas as pd

def add_percentiles_wide(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_cols: list[str],
    prefix: str = "pct_",
) -> pd.DataFrame:
    out = df.copy()
    for col in metric_cols:
        out[f"{prefix}{col}"] = out.groupby(group_cols, dropna=False)[col].transform(
            lambda s: s.rank(pct=True, method="average") * 100
        )
    return out

def build_percentiles_long(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_cols: list[str],
    pct_scope: str,
    id_cols: list[str],
) -> pd.DataFrame:
    """
    Returns long-form percentiles:
      id cols + kpi_name + kpi_value + kpi_pct + pct_scope
    """
    tmp = df[id_cols + metric_cols].copy()

    # compute pct columns in a temp wide frame
    tmp2 = add_percentiles_wide(tmp, metric_cols=metric_cols, group_cols=group_cols, prefix="pct_")

    # melt into long
    val_long = tmp2.melt(
        id_vars=id_cols,
        value_vars=metric_cols,
        var_name="kpi_name",
        value_name="kpi_value",
    )
    pct_long = tmp2.melt(
        id_vars=id_cols,
        value_vars=[f"pct_{m}" for m in metric_cols],
        var_name="kpi_name_pct",
        value_name="kpi_pct",
    )
    pct_long["kpi_name"] = pct_long["kpi_name_pct"].str.replace("^pct_", "", regex=True)
    pct_long = pct_long.drop(columns=["kpi_name_pct"])

    out = val_long.merge(pct_long, on=id_cols + ["kpi_name"], how="left", validate="1:1")
    out["pct_scope"] = pct_scope
    return out

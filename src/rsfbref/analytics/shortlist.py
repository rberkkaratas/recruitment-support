from __future__ import annotations
import pandas as pd

SUBSCORES = {
    "BPCB": {
        "progression": ["pct_prog_passes_p90", "pct_passes_final_third_p90", "pct_prog_carries_p90"],
        "defending": ["pct_tkl_int_p90", "pct_clr_p90", "pct_aerial_win_pct"],
        "security": ["pct_pass_cmp_pct", "pct_errors_p90"],
        "creation": [],  # not primary
    },
    "DLP": {
        "progression": ["pct_prog_passes_p90", "pct_passes_final_third_p90", "pct_prog_carries_p90"],
        "creation": ["pct_key_passes_p90", "pct_xa_p90", "pct_sca_p90"],
        "defending": ["pct_tkl_int_p90"],
        "security": ["pct_pass_cmp_pct", "pct_fouls_p90"],
    },
    "WCR": {
        "progression": ["pct_prog_carries_p90", "pct_carries_pa_p90", "pct_succ_takeons_p90"],
        "creation": ["pct_xa_p90", "pct_key_passes_p90", "pct_sca_p90", "pct_crosses_pa_p90"],
        "finishing": ["pct_Per_90_Minutes_npxG"],
        "security": ["pct_takeon_succ_pct", "pct_mis_dis_p90"],
    },
}

def _mean_pct(df: pd.DataFrame, cols: list[str], invert: list[str] | None = None) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.Series(pd.NA, index=df.index)
    X = df[cols].astype(float).copy()
    if invert:
        for c in invert:
            if c in X.columns:
                X[c] = 100 - X[c]
    return X.mean(axis=1)

def _evidence_strings(df: pd.DataFrame, role_id: str, top_k: int = 5) -> pd.DataFrame:
    # choose evidence KPIs per role (raw + pct)
    role_kpis = {
        "BPCB": ["prog_passes_p90", "passes_final_third_p90", "pass_cmp_pct", "tkl_int_p90", "aerial_win_pct"],
        "DLP": ["passes_att_p90", "prog_passes_p90", "pass_cmp_pct", "xa_p90", "sca_p90"],
        "WCR": ["prog_carries_p90", "succ_takeons_p90", "xa_p90", "sca_p90", "Per_90_Minutes_npxG"],
    }[role_id]

    out = df.copy()
    for i, k in enumerate(role_kpis[:top_k], start=1):
        pct = f"pct_{k}"
        if k in out.columns and pct in out.columns:
            out[f"evidence_{i}"] = out.apply(lambda r: f"{k}={r[k]:.2f} (p{r[pct]:.0f})" if pd.notna(r[k]) and pd.notna(r[pct]) else None, axis=1)
        else:
            out[f"evidence_{i}"] = None
    return out

def build_shortlist(df: pd.DataFrame, role_id: str, top_n: int = 50) -> pd.DataFrame:
    score_col = f"score_{role_id}"
    pool = df[df[score_col].notna()].copy()
    if pool.empty:
        return pd.DataFrame()

    # Subscores (with transparent inversions where relevant)
    if role_id == "BPCB":
        pool["sub_security"] = _mean_pct(pool, ["pct_pass_cmp_pct", "pct_errors_p90"], invert=["pct_errors_p90"])
    elif role_id == "DLP":
        pool["sub_security"] = _mean_pct(pool, ["pct_pass_cmp_pct", "pct_fouls_p90"], invert=["pct_fouls_p90"])
    else:
        pool["sub_security"] = _mean_pct(pool, ["pct_takeon_succ_pct", "pct_mis_dis_p90"], invert=["pct_mis_dis_p90"])

    for sub, cols in SUBSCORES[role_id].items():
        if sub == "security":
            continue
        pool[f"sub_{sub}"] = _mean_pct(pool, cols)

    pool["total_score"] = pool[score_col]

    # Risk flags (v1, rule-based)
    flags = []
    flags.append(pool["minutes"].between(900, 1200).map(lambda x: "LOW_MINUTES" if x else None))
    if "pct_errors_p90" in pool.columns:
        flags.append((pool["pct_errors_p90"] >= 90).map(lambda x: "HIGH_ERRORS" if x else None))
    if "pct_mis_dis_p90" in pool.columns:
        flags.append((pool["pct_mis_dis_p90"] >= 90).map(lambda x: "HIGH_TURNOVERS" if x else None))
    flags.append((pool["age"] <= 19).map(lambda x: "VERY_YOUNG" if x else None))
    flags.append((pool["age"] >= 32).map(lambda x: "OLDER_PROFILE" if x else None))

    flags_df = pd.concat(flags, axis=1)
    pool["risk_flags"] = flags_df.apply(lambda r: "|".join([x for x in r.tolist() if isinstance(x, str)]) if any(isinstance(x, str) for x in r.tolist()) else "", axis=1)
    pool["risk_count"] = pool["risk_flags"].apply(lambda s: 0 if s == "" else len(s.split("|")))

    # Evidence
    pool = _evidence_strings(pool, role_id=role_id, top_k=5)

    # Final output columns (Tableau-friendly)
    out_cols = [
        "player_id", "team_id", "league", "season", "position_bucket", "minutes",
        "total_score",
        "sub_progression", "sub_defending", "sub_creation", "sub_finishing", "sub_security",
        "risk_flags", "risk_count",
        "evidence_1", "evidence_2", "evidence_3", "evidence_4", "evidence_5",
    ]
    out_cols = [c for c in out_cols if c in pool.columns]

    out = pool[out_cols].copy()
    out["role_id"] = role_id

    out = out.sort_values(["total_score", "minutes"], ascending=[False, False]).head(top_n)
    out["rank"] = range(1, len(out) + 1)
    return out

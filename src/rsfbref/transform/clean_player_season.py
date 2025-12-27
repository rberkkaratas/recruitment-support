from __future__ import annotations
import pandas as pd

KEY = ["league", "season", "team", "player"]

def _to_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def infer_position_bucket(pos: str) -> str:
    if pos is None or pd.isna(pos):
        return "UNK"
    p = str(pos)

    if "GK" in p:
        return "GK"

    # WIDE: any forward who is not a pure CF bucket in our v1
    # (FBref doesn't give winger explicitly here, so we allow FW and FW,MF)
    if "FW" in p:
        return "WIDE"

    # DM/CM: any MF without FW
    if "MF" in p and "FW" not in p:
        return "DMCM"

    # CB: DF without FW
    if "DF" in p and "FW" not in p:
        return "CB"

    return "OTHER"
def build_player_season_clean(base: pd.DataFrame, min_minutes: int = 900) -> pd.DataFrame:
    df = base.copy()

    # Core time fields (you have both; we standardize to one)
    df = _to_numeric(df, ["Playing_Time_Min", "Playing_Time_90s"])
    df["minutes"] = df["Playing_Time_Min"]
    df["nineties"] = df["Playing_Time_90s"]

    # Buckets
    df["position_bucket"] = df["pos"].apply(infer_position_bucket)

    # Exclude GK for v1 recruitment roles
    df = df[df["position_bucket"] != "GK"].copy()

    # Minutes filter
    df = df[df["minutes"] >= min_minutes].copy()

    # Helper: per90 from totals
    def per90(total_col: str) -> pd.Series:
        return pd.to_numeric(df[total_col], errors="coerce") / df["nineties"]

    # ---- Canonical passing/progression features ----
    df["pass_cmp_pct"] = pd.to_numeric(df["passing__Total_Cmppct"], errors="coerce")
    df["passes_att_p90"] = per90("passing__Total_Att")
    df["prog_passes_p90"] = per90("passing__PrgP")
    df["passes_final_third_p90"] = per90("passing__1/3")
    df["long_pass_cmp_pct"] = pd.to_numeric(df["passing__Long_Cmppct"], errors="coerce")
    df["key_passes_p90"] = per90("passing__KP")
    df["xa_p90"] = per90("passing__Expected_xA")  # expected assists (total) -> per90
    df["crosses_pa_p90"] = per90("passing__CrsPA")

    # ---- Defending/duels ----
    df["tkl_int_p90"] = per90("defense__Tkl+Int")
    df["clr_p90"] = per90("defense__Clr")
    df["errors_p90"] = per90("defense__Err")
    df["aerial_win_pct"] = pd.to_numeric(df["misc__Aerial_Duels_Wonpct"], errors="coerce")

    # ---- Carrying/dribbling ----
    df["prog_carries_p90"] = per90("possession__Carries_PrgC")
    df["carries_pa_p90"] = per90("possession__Carries_CPA")         # carries into pen area
    df["succ_takeons_p90"] = per90("possession__Take-Ons_Succ")
    df["takeon_succ_pct"] = pd.to_numeric(df["possession__Take-Ons_Succpct"], errors="coerce")

    # ---- Creation aggregate ----
    df["sca_p90"] = pd.to_numeric(df["goal_shot_creation__SCA_SCA90"], errors="coerce")

    # ---- Negatives / risk ----
    df["mis_dis_p90"] = (per90("possession__Carries_Mis") + per90("possession__Carries_Dis"))
    df["fouls_p90"] = per90("misc__Performance_Fls")

    # Keep only what we need + identity columns (for marts)
    keep = KEY + [
        "nation", "pos", "age", "born",
        "minutes", "nineties", "position_bucket",
        # canonical metrics
        "pass_cmp_pct", "passes_att_p90", "prog_passes_p90", "passes_final_third_p90",
        "long_pass_cmp_pct", "key_passes_p90", "xa_p90", "crosses_pa_p90",
        "tkl_int_p90", "clr_p90", "errors_p90", "aerial_win_pct",
        "prog_carries_p90", "carries_pa_p90", "succ_takeons_p90", "takeon_succ_pct",
        "sca_p90", "mis_dis_p90", "fouls_p90",
        # handy built-ins (already per90 in your base)
        "Per_90_Minutes_npxG",
    ]
    keep = [c for c in keep if c in df.columns]
    return df[keep].copy()

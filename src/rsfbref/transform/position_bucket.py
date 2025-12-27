from __future__ import annotations

from pathlib import Path
import yaml
import pandas as pd


def load_position_map(path: str | Path) -> dict:
    p = Path(path)
    return yaml.safe_load(p.read_text(encoding="utf-8"))


def normalize_pos(pos: object) -> str:
    if pos is None or (isinstance(pos, float) and pd.isna(pos)) or pd.isna(pos):
        return ""
    s = str(pos).strip()
    # standardize delimiter formatting (FBref uses commas)
    s = s.replace(" ", "")
    return s


def _is_num(x: object) -> bool:
    try:
        return pd.notna(x) and float(x) == float(x)
    except Exception:
        return False


def _f(x: object) -> float | None:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None


def infer_position_bucket(
    pos_raw: object,
    *,
    crosses_pa_p90: object = None,
    xa_p90: object = None,
    aerial_win_pct: object = None,
    clr_p90: object = None,
    config: dict,
) -> tuple[str, str]:
    """
    Returns (position_bucket, reason_code).

    Buckets: GK, CB, FB, DMCM, WIDE, CF, OTHER
    """
    pos = normalize_pos(pos_raw)

    # Unconfigured fallback
    rules = (config or {}).get("rules", {})
    maps = (config or {}).get("mappings", {})
    pos_exact = maps.get("pos_exact", {})
    pos_combo = maps.get("pos_combo", {})

    # Base classification by normalized pos string
    base = pos_exact.get(pos) or pos_combo.get(pos) or ""

    # If pos includes GK anywhere, treat as GK
    if "GK" in pos:
        return "GK", "POS_HAS_GK"

    # --- Forwards / wide ---
    # FBref does not separate winger/striker well; we keep it explainable:
    # FW only -> CF
    # MF+FW combo -> WIDE (our scouting lens for creator-wingers)
    if base == "FW":
        return "CF", "POS_FW_ONLY"
    if base in {"MFFW", "FWMF"}:
        return "WIDE", "POS_MF_FW"

    # --- Midfield ---
    # MF only -> DMCM (we group DM/CM for now)
    # MF,DF -> DMCM (defensive mid types)
    if base == "MF":
        return "DMCM", "POS_MF_ONLY"
    if base == "MFDF":
        return "DMCM", "POS_MF_DF"

    # --- Defense: DF and DF,MF need heuristics to avoid CB/FB leakage ---
    df_split = rules.get("df_split", {})
    dfmf_split = rules.get("df_mf_split", {})

    x_cross = _f(crosses_pa_p90)
    x_xa = _f(xa_p90)
    x_aer = _f(aerial_win_pct)
    x_clr = _f(clr_p90)

    if base == "DF":
        # CB strong signals
        if x_aer is not None and x_aer >= float(df_split.get("cb_if_aerial_win_pct_gte", 55.0)):
            return "CB", "DF_CB_AERIAL"
        if x_clr is not None and x_clr >= float(df_split.get("cb_if_clr_p90_gte", 4.0)):
            return "CB", "DF_CB_CLEARANCES"

        # FB strong signals
        if x_cross is not None and x_cross >= float(df_split.get("fb_if_crosses_pa_p90_gte", 1.2)):
            return "FB", "DF_FB_CROSSES"
        if x_xa is not None and x_xa >= float(df_split.get("fb_if_xa_p90_gte", 0.08)):
            return "FB", "DF_FB_XA"

        # default
        return str(df_split.get("default_df_bucket", "CB")), "DF_DEFAULT"

    if base == "DFMF":
        # FB signals
        if x_cross is not None and x_cross >= float(dfmf_split.get("fb_if_crosses_pa_p90_gte", 1.0)):
            return "FB", "DFMF_FB_CROSSES"
        if x_xa is not None and x_xa >= float(dfmf_split.get("fb_if_xa_p90_gte", 0.06)):
            return "FB", "DFMF_FB_XA"

        return str(dfmf_split.get("default_bucket", "DMCM")), "DFMF_DEFAULT"

    # Anything else -> OTHER
    if base:
        return "OTHER", f"POS_{base}"
    return "OTHER", "POS_UNKNOWN"

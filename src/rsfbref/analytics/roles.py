from __future__ import annotations
import yaml
import pandas as pd

def load_roles(path: str) -> list[dict]:
    return yaml.safe_load(open(path, "r", encoding="utf-8"))["roles"]

def apply_must_haves(df: pd.DataFrame, role: dict) -> pd.Series:
    mh = role.get("must_have", {})
    mask = pd.Series(True, index=df.index)

    if "min_minutes" in mh:
        mask &= df["minutes"] >= mh["min_minutes"]

    # Map must-have fields to canonical columns
    # (these names correspond to your roles_v1.yaml keys)
    if "pass_cmp_pct_min" in mh:
        mask &= df["pass_cmp_pct"] >= mh["pass_cmp_pct_min"]
    if "prog_passes_p90_min" in mh:
        mask &= df["prog_passes_p90"] >= mh["prog_passes_p90_min"]
    if "aerial_win_pct_min" in mh and "aerial_win_pct" in df.columns:
        mask &= df["aerial_win_pct"] >= mh["aerial_win_pct_min"]
    if "passes_att_p90_min" in mh:
        mask &= df["passes_att_p90"] >= mh["passes_att_p90_min"]
    if "prog_carries_p90_min" in mh:
        mask &= df["prog_carries_p90"] >= mh["prog_carries_p90_min"]
    if "succ_takeons_p90_min" in mh:
        mask &= df["succ_takeons_p90"] >= mh["succ_takeons_p90_min"]
    if "xa_p90_min" in mh:
        mask &= df["xa_p90"] >= mh["xa_p90_min"]

    return mask

def score_roles(df: pd.DataFrame, roles_yaml_path: str) -> pd.DataFrame:
    out = df.copy()
    roles = load_roles(roles_yaml_path)

    for role in roles:
        role_id = role["role_id"]
        bucket = role["position_bucket"]
        weights = role["weights"]
        negatives = set(role.get("negative_metrics", []))

        # eligibility by bucket
        elig = out["position_bucket"].eq(bucket)

        # must-haves
        elig &= apply_must_haves(out, role)

        # build weighted score from percentiles
        score = pd.Series(0.0, index=out.index)
        wsum = 0.0

        for k, w in weights.items():
            # map config keys -> canonical percentile columns
            # we encode “combined” keys in config; resolve them here.
            if k == "long_pass_cmp_p90_or_pct":
                feat = "long_pass_cmp_pct"
            elif k == "aerial_win_pct_or_won_p90":
                feat = "aerial_win_pct"
            elif k == "errors_or_dispossessed_neg":
                feat = "errors_p90"  # we treat errors as the negative proxy for BPCB
            elif k == "dispossessed_miscontrols_neg":
                feat = "mis_dis_p90"
            elif k == "fouls_committed_neg":
                feat = "fouls_p90"
            else:
                feat = k

            pct_col = f"pct_{feat}"
            if pct_col not in out.columns:
                # If a feature is missing, skip it (v1 robustness)
                continue

            val = out[pct_col]
            if k in negatives:
                val = 100 - val

            score += w * val
            wsum += w

        out[f"score_{role_id}"] = (score / wsum).where(elig, pd.NA)

    return out

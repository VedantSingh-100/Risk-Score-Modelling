# metrics.py
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

def gini_from_auc(auc: float) -> float:
    return 2.0 * auc - 1.0

def ks_stat(y_true, y_score):
    df = pd.DataFrame({"y": y_true, "p": y_score}).sort_values("p")
    df["cum_pos"] = (df["y"]==1).cumsum() / (df["y"]==1).sum()
    df["cum_neg"] = (df["y"]==0).cumsum() / (df["y"]==0).sum()
    return float(np.max(np.abs(df["cum_pos"] - df["cum_neg"])))

def decile_table(y_true, y_score, n_bins=10):
    df = pd.DataFrame({"y": y_true, "p": y_score})
    # robust deciling on prob with ties
    df["decile"] = pd.qcut(df["p"].rank(method="first"), q=n_bins, labels=list(range(n_bins,0,-1)))
    agg = df.groupby("decile").agg(
        n=("y","size"),
        pos=("y","sum"),
        avg_p=("p","mean")
    ).reset_index()
    agg["neg"] = agg["n"] - agg["pos"]
    agg["pos_rate"] = agg["pos"] / agg["n"]
    agg["cum_pos"] = agg["pos"].cumsum()
    agg["cum_neg"] = agg["neg"].cumsum()
    total_pos, total_neg = agg["pos"].sum(), agg["neg"].sum()
    agg["cum_pos_capture"] = agg["cum_pos"] / total_pos
    agg["cum_neg_share"] = agg["cum_neg"] / (total_neg if total_neg>0 else 1.0)
    # classic lift = pos_rate / overall_rate
    overall_rate = total_pos / (total_pos + total_neg)
    agg["lift"] = agg["pos_rate"] / overall_rate
    return agg

def summarize_all(y_true, y_score, label="oof", n_bins=10):
    auc = roc_auc_score(y_true, y_score)
    ap  = average_precision_score(y_true, y_score)
    gini = gini_from_auc(auc)
    ks = ks_stat(y_true, y_score)
    dec = decile_table(y_true, y_score, n_bins=n_bins)
    summary = {
        "label": label,
        "auc": float(auc),
        "ap":  float(ap),
        "gini": float(gini),
        "ks": float(ks)
    }
    return summary, dec


#Plots.py

import os
import math
import matplotlib.pyplot as plt
import pandas as pd

def _ensure_dir(path: str = "figs"):
    os.makedirs(path, exist_ok=True)
    return path

def _ci_to_err(low, high, mean):
    return (mean - low, high - mean)


def plot_metric_vs_lambda(df_all: pd.DataFrame, metric: str, ylabel: str, title: str, outname:str):

    mean_col = f"{metric}_mean"
    lo_col = f"{metric}_ci_low"
    hi_col = f"{metric}_ci_high"

    miss = [c for c in [mean_col, lo_col, hi_col, "lam", "scenario"] if c not in df_all.columns]
    if miss:
        raise ValueError(f"faltan columnas en df: {miss}")
    
    fig, ax = plt.subplots(figsize = (7,4))
    for scen, g in df_all.groupby("scenario"):
        g = g.sort_values("lam")
        y = g[mean_col].values
        lo = g[lo_col].values
        hi = g[hi_col].values
        yerr = _ci_to_err(lo, hi, y)
        ax.errorbar(g["lam"].values, y, yerr = yerr, fmt='-o', label=scen, capsize = 3)

    ax.set_xlabel("λ (arribos por minuto)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    outdir = _ensure_dir()
    fig.savefig(os.path.join(outdir, f"{outname}.png"), dpi=200)
    return fig, ax



from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr

def _finite_pair(x: np.ndarray, y: np.ndarray):
    ok = np.isfinite(x) & np.isfinite(y)
    return x[ok], y[ok], int(ok.sum())

def spearman_vec(x: np.ndarray, y: np.ndarray, *, n_min: int = 10) -> float:
    x2, y2, n = _finite_pair(x, y)
    if n < n_min:
        return np.nan
    return float(spearmanr(x2, y2).correlation)

def pearson_zscore_vec(x: np.ndarray, y: np.ndarray, *, n_min: int = 10) -> float:
    x2, y2, n = _finite_pair(x, y)
    if n < n_min:
        return np.nan
    sx = np.nanstd(x2)
    sy = np.nanstd(y2)
    if sx == 0 or sy == 0:
        return np.nan
    zx = (x2 - np.nanmean(x2)) / sx
    zy = (y2 - np.nanmean(y2)) / sy
    return float(np.corrcoef(zx, zy)[0, 1])

def rmse_robust_iqr_vec(x: np.ndarray, y: np.ndarray, *, n_min: int = 10) -> float:
    x2, y2, n = _finite_pair(x, y)
    if n < n_min:
        return np.nan

    def rnorm(v):
        med = np.nanmedian(v)
        q75, q25 = np.nanpercentile(v, [75, 25])
        iqr = q75 - q25
        if iqr == 0:
            return np.full_like(v, np.nan)
        return (v - med) / iqr

    xn = rnorm(x2)
    yn = rnorm(y2)
    ok = np.isfinite(xn) & np.isfinite(yn)
    if ok.sum() < n_min:
        return np.nan
    return float(np.sqrt(np.nanmean((xn[ok] - yn[ok]) ** 2)))
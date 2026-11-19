"""Evaluation and material for the slides (step 9).

All losses are computed on the *variance* scale (average daily GK variance over
the horizon), i.e. on the realised target exp(target_h).
  QLIKE = mean( rv/f - log(rv/f) - 1 )      (robust to noise in the proxy)
  MSE   = mean( (rv - f)^2 )
"""
from math import erf, sqrt

import numpy as np
import pandas as pd

import config


def qlike(rv, f) -> np.ndarray:
    r = np.asarray(rv) / np.asarray(f)
    return r - np.log(r) - 1


def sq_err(rv, f) -> np.ndarray:
    return (np.asarray(rv) - np.asarray(f)) ** 2


def diebold_mariano(loss_a, loss_b, h: int) -> tuple[float, float]:
    """DM test of equal predictive accuracy (Newey-West variance, lag h-1).
    Negative statistic => model A has lower loss. Returns (stat, two-sided p)."""
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n, dbar = len(d), d.mean()
    dc = d - dbar
    var = dc @ dc / n
    for lag in range(1, h):
        var += 2 * (1 - lag / h) * (dc[lag:] @ dc[:-lag]) / n
    stat = dbar / np.sqrt(var / n)
    p = 2 * (1 - 0.5 * (1 + erf(abs(stat) / sqrt(2))))
    return float(stat), float(p)


def loss_table(rv: pd.Series, forecasts: dict[str, pd.Series], h: int,
               crisis: pd.Series) -> pd.DataFrame:
    """QLIKE / MSE per model, overall and by regime, plus DM vs HAR."""
    rows = []
    for regime, mask in [("all", np.ones(len(rv), bool)), ("calm", ~crisis.to_numpy()),
                         ("crisis", crisis.to_numpy())]:
        if mask.sum() == 0:
            continue
        for name, f in forecasts.items():
            q, m = qlike(rv[mask], f[mask]), sq_err(rv[mask], f[mask])
            row = {"horizon": h, "regime": regime, "model": name, "n": int(mask.sum()),
                   "QLIKE": q.mean(), "MSE": m.mean()}
            if name != "HAR" and "HAR" in forecasts:
                hq, hm = qlike(rv[mask], forecasts["HAR"][mask]), sq_err(rv[mask], forecasts["HAR"][mask])
                row["QLIKE_vs_HAR"] = q.mean() / hq.mean()
                row["MSE_vs_HAR"] = m.mean() / hm.mean()
                row["DM_QLIKE_stat"], row["DM_QLIKE_p"] = diebold_mariano(q, hq, h)
            rows.append(row)
    return pd.DataFrame(rows)


def seed_stability(rv: pd.Series, per_seed: pd.DataFrame, s2: float, h: int) -> pd.DataFrame:
    """Test QLIKE of each individual seed vs the seed average."""
    adj = np.exp(0.5 * s2) if config.BIAS_CORRECTION else 1.0
    rows = [{"horizon": h, "member": c, "QLIKE": qlike(rv, np.exp(per_seed[c]) * adj).mean()}
            for c in per_seed.columns]
    rows.append({"horizon": h, "member": "ensemble",
                 "QLIKE": qlike(rv, np.exp(per_seed.mean(axis=1)) * adj).mean()})
    out = pd.DataFrame(rows)
    singles = out[out.member != "ensemble"].QLIKE
    out["seed_mean"], out["seed_std"] = singles.mean(), singles.std()
    return out


def permutation_importance(ens, data: pd.DataFrame, positions: np.ndarray,
                           n_repeats: int = 3, seed: int = 0) -> pd.DataFrame:
    """Increase in validation MSE(log) when one feature is shuffled in time.
    Uses validation data only (no peeking at the test set)."""
    rng = np.random.default_rng(seed)
    y = data[f"target_{ens.h}"].to_numpy()[positions]
    base_x = ens.prep.x_scaled
    base = np.mean((ens.predict_per_seed(data, positions).mean(0) - y) ** 2)
    rows = []
    for j, col in enumerate(ens.feature_cols):
        incs = []
        for _ in range(n_repeats):
            x = base_x.copy()
            x[:, j] = rng.permutation(x[:, j])
            pred = ens.predict_per_seed(data, positions, x_scaled=x).mean(0)
            incs.append(np.mean((pred - y) ** 2) - base)
        rows.append({"horizon": ens.h, "feature": col, "mse_increase": float(np.mean(incs)),
                     "relative_increase": float(np.mean(incs) / base)})
    return pd.DataFrame(rows).sort_values("mse_increase", ascending=False)


def crisis_flag(data: pd.DataFrame, dates: pd.DatetimeIndex, train_end) -> pd.Series:
    """Ex-ante regime: lagged monthly log variance above its training quantile."""
    thr = data.loc[:train_end, "har_m"].quantile(config.CRISIS_QUANTILE)
    return (data.loc[dates, "har_m"] > thr).rename("crisis")

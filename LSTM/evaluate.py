"""Evaluation and material for the slides (step 9).

All losses are computed on the *variance* scale, against `actual` from the
phase's actuals.csv (average daily realized variance over the next h sessions).
  RMSE  = sqrt(mean( (rv - f)^2 ))          (the group's primary metric)
  QLIKE = mean( rv/f - log(rv/f) - 1 )      (secondary, robust to proxy noise)
Every model is scored on the same origins, together with the persistence
`baseline` of actuals.csv.
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
    """RMSE / QLIKE per model, overall and by regime, with ratios and
    Diebold-Mariano tests against the HAR benchmark (model_3's har_rv) and
    against the persistence baseline."""
    rows = []
    for regime, mask in [("all", np.ones(len(rv), bool)), ("calm", ~crisis.to_numpy()),
                         ("crisis", crisis.to_numpy())]:
        if mask.sum() == 0:
            continue
        for name, f in forecasts.items():
            q, m = qlike(rv[mask], f[mask]), sq_err(rv[mask], f[mask])
            row = {"horizon": h, "regime": regime, "model": name, "n": int(mask.sum()),
                   "RMSE": np.sqrt(m.mean()), "QLIKE": q.mean(), "MSE": m.mean()}
            for ref, tag in [(config.BENCHMARK_NAME, "HAR"), ("persistence", "pers")]:
                if name == ref or ref not in forecasts:
                    continue
                rq, rm = qlike(rv[mask], forecasts[ref][mask]), sq_err(rv[mask], forecasts[ref][mask])
                row[f"RMSE_vs_{tag}"] = np.sqrt(m.mean() / rm.mean())
                row[f"QLIKE_vs_{tag}"] = q.mean() / rq.mean()
                row[f"DM_SE_{tag}_p"] = diebold_mariano(m, rm, h)[1]
                row[f"DM_QLIKE_{tag}_p"] = diebold_mariano(q, rq, h)[1]
            rows.append(row)
    return pd.DataFrame(rows)


def seed_stability(rv: pd.Series, per_seed: pd.DataFrame, s2: float, h: int,
                   har: pd.Series, crisis: pd.Series) -> pd.DataFrame:
    """Test RMSE and QLIKE of each individual seed vs the seed average, per
    regime, with ratios to the HAR benchmark (< 1 means that seed beats it)."""
    adj = np.exp(0.5 * s2) if config.BIAS_CORRECTION else 1.0
    members = {c: np.exp(per_seed[c]) * adj for c in per_seed.columns}
    members["ensemble"] = np.exp(per_seed.mean(axis=1)) * adj
    crisis = crisis.to_numpy()
    rows = []
    for regime, mask in [("all", np.ones(len(rv), bool)), ("calm", ~crisis), ("crisis", crisis)]:
        har_q = qlike(rv[mask], har[mask]).mean()
        har_r = np.sqrt(sq_err(rv[mask], har[mask]).mean())
        for name, fc in members.items():
            q = qlike(rv[mask], fc[mask]).mean()
            r = np.sqrt(sq_err(rv[mask], fc[mask]).mean())
            rows.append({"horizon": h, "regime": regime, "member": name, "RMSE": r,
                         "RMSE_vs_HAR": r / har_r, "QLIKE": q, "QLIKE_vs_HAR": q / har_q})
    out = pd.DataFrame(rows)
    singles = out[out.member != "ensemble"].groupby("regime")["QLIKE_vs_HAR"]
    out["ratio_seed_mean"] = out.regime.map(singles.mean())
    out["ratio_seed_std"] = out.regime.map(singles.std())
    out["ratio_seed_min"] = out.regime.map(singles.min())
    out["ratio_seed_max"] = out.regime.map(singles.max())
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

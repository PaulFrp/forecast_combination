"""Targets (step 1) and features (step 2).

Timing convention, used everywhere in this folder
-------------------------------------------------
A row dated t is a forecast made at the close of t-1 for the period starting at t.
  * every feature on row t uses data up to t-1 only (hence the .shift(1)),
  * target_h on row t = log( mean(gk_var[t], ..., gk_var[t+h-1]) ),
    i.e. the log of the average daily variance over the next h days.
The exported `date` is therefore the first day of the forecast window.
"""
import numpy as np
import pandas as pd

import config


def build_targets(gk_var: pd.Series, horizons=config.HORIZONS) -> pd.DataFrame:
    """log of the average GK variance over days t .. t+h-1 (NaN near the end)."""
    out = {}
    for h in horizons:
        fwd_mean = gk_var.rolling(h).mean().shift(-(h - 1))
        out[f"target_{h}"] = np.log(fwd_mean)
    return pd.DataFrame(out, index=gk_var.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features dated t, using information up to t-1."""
    f = pd.DataFrame(index=df.index)
    var_lag = df["gk_var"].shift(1)

    # HAR components (Corsi 2009): average the *variance* over 1/5/22 days, then log.
    for name, w in config.HAR_WINDOWS.items():
        f[name] = np.log(var_lag.rolling(w).mean())

    # Returns: last day's log return, its negative part (leverage effect:
    # falls raise volatility more than rises) and last week's cumulative return.
    ret = np.log(df["Close"]).diff()
    f["ret_1"] = ret.shift(1)
    f["neg_ret_1"] = ret.clip(upper=0).shift(1)
    f["ret_5"] = ret.rolling(5).sum().shift(1)

    # Volume: log volume trends up over 20 years, so we use it relative to its
    # own past 66-day average (otherwise the test period lies outside the range
    # the scaler saw in training).
    log_vol = np.log(df["Volume"].replace(0, np.nan)).ffill()
    f["log_volume"] = (log_vol - log_vol.rolling(66).mean()).shift(1)

    # Market-implied volatility (forward-looking information).
    if "vix" in df:
        f["log_vix"] = np.log(df["vix"]).shift(1)

    return f


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Features + targets in one frame; rows with missing features dropped
    (the warm-up period at the very start). Targets may still be NaN at the end."""
    feats = build_features(df)
    targets = build_targets(df["gk_var"])
    data = pd.concat([feats, targets, df["gk_var"]], axis=1)
    data = data.dropna(subset=list(feats.columns))
    return data, list(feats.columns)

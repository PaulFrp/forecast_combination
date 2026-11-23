"""Targets (step 1) and features (step 2).

Timing convention (group contract), used everywhere in this folder
------------------------------------------------------------------
A row dated t is a forecast *origin*: the forecast is made at 17:00 New York
time on session t, after that session has closed.
  * every feature on row t uses data through session t only,
  * target_h on row t = log( mean(rv[t+1], ..., rv[t+h]) ),
    i.e. the log of the average daily realized variance over the next h
    sessions, excluding the origin (= log of `actual` in actuals.csv).
"""
import numpy as np
import pandas as pd

import config


def forward_mean(rv: pd.Series, h: int) -> pd.Series:
    """mean(rv[t+1 .. t+h]) on row t (NaN for the last h rows)."""
    return rv.rolling(h).mean().shift(-h)


def build_targets(rv: pd.Series, horizons=config.HORIZONS) -> pd.DataFrame:
    """log of the average RV over sessions t+1 .. t+h."""
    rv = rv.clip(lower=config.RV_FLOOR)
    return pd.DataFrame({f"target_{h}": np.log(forward_mean(rv, h)) for h in horizons}, index=rv.index)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """All features dated t, using information through session t."""
    f = pd.DataFrame(index=df.index)
    rv = df["spy_rv"].clip(lower=config.RV_FLOOR)

    # HAR components (Corsi 2009): average the *variance* over 1/5/22 sessions, then log.
    for name, w in config.HAR_WINDOWS.items():
        f[name] = np.log(rv.rolling(w).mean())

    # Returns: the session's close-to-close log return, its negative part
    # (leverage effect: falls raise volatility more than rises) and the
    # cumulative return over the last 5 sessions.
    ret = np.log(df["spy_close"]).diff()
    f["ret_1"] = ret
    f["neg_ret_1"] = ret.clip(upper=0)
    f["ret_5"] = ret.rolling(5).sum()

    # Volume relative to its own past 66-session average (raw volume trends).
    log_vol = np.log(df["spy_volume"].replace(0, np.nan)).ffill()
    f["log_volume"] = log_vol - log_vol.rolling(66).mean()

    # Market-implied volatility, lagged because of its unknown publication time.
    if config.USE_VIX and "vix_close" in df:
        f["log_vix"] = np.log(df["vix_close"]).shift(config.VIX_LAG)

    return f


def build_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Features + targets + spy_rv in one frame; warm-up rows with missing
    features dropped. Targets are NaN for the last h sessions."""
    feats = build_features(df)
    targets = build_targets(df["spy_rv"])
    data = pd.concat([feats, targets, df["spy_rv"]], axis=1)
    data = data.dropna(subset=list(feats.columns))
    return data, list(feats.columns)

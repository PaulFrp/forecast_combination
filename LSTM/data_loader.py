"""Load prices, compute Garman-Klass variance, fetch VIX, read the group split."""
import warnings

import numpy as np
import pandas as pd

import config


def load_prices() -> pd.DataFrame:
    """Daily S&P 500 OHLCV indexed by date, sorted, incomplete rows dropped."""
    df = pd.read_csv(config.FULL_DATA, parse_dates=["Date"], index_col="Date")
    df = df.sort_index().dropna()
    df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]
    return df


def garman_klass_variance(df: pd.DataFrame) -> pd.Series:
    """Daily Garman-Klass (1980) variance estimator.

    sigma^2 = 0.5 * ln(H/L)^2 - (2 ln 2 - 1) * ln(C/O)^2
    Floored at config.GK_FLOOR so that logs are always defined.
    """
    hl = np.log(df["High"] / df["Low"])
    co = np.log(df["Close"] / df["Open"])
    gk = 0.5 * hl**2 - (2 * np.log(2) - 1) * co**2
    return gk.clip(lower=config.GK_FLOOR).rename("gk_var")


def load_vix(index: pd.DatetimeIndex) -> pd.Series | None:
    """VIX close aligned on `index` (forward-filled). Downloaded once and cached
    in data/vix_daily.csv so every later run is offline and reproducible."""
    if not config.VIX_CACHE.exists():
        try:
            import yfinance as yf
            raw = yf.download(config.VIX_TICKER, start=index.min() - pd.Timedelta(days=10),
                              end=index.max() + pd.Timedelta(days=2),
                              auto_adjust=False, progress=False)
            raw.columns = raw.columns.get_level_values(0)
            if raw.empty:
                raise RuntimeError("empty download")
            raw[["Close"]].rename(columns={"Close": "VIX"}).to_csv(config.VIX_CACHE)
        except Exception as exc:  # no internet, yfinance missing, ...
            warnings.warn(f"Could not download VIX ({exc}); continuing without it.")
            return None
    vix = pd.read_csv(config.VIX_CACHE, parse_dates=["Date"], index_col="Date")["VIX"]
    vix = vix[~vix.index.duplicated()].sort_index()
    return vix.reindex(index).ffill().rename("vix")


def load_split_dates() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """(first_date, last_date) of each group split file."""
    out = {}
    for name, path in config.SPLIT_FILES.items():
        dates = pd.read_csv(path, usecols=["Date"], parse_dates=["Date"])["Date"]
        out[name] = (dates.min(), dates.max())
    # Sanity: chronological and non-overlapping.
    assert out["train"][1] < out["val"][0] <= out["val"][1] < out["test"][0], out
    return out


def load_all() -> tuple[pd.DataFrame, dict]:
    """Prices + gk_var (+ vix) in one frame, and the split date ranges."""
    df = load_prices()
    df["gk_var"] = garman_klass_variance(df)
    if config.USE_VIX:
        vix = load_vix(df.index)
        if vix is not None:
            df["vix"] = vix
    return df, load_split_dates()

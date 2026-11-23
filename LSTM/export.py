"""Back-transform log forecasts and export them in the group contract (step 8).

Contract: one CSV with columns  date, horizon, forecast, model
  date     : first day of the forecast window (forecast made at the close of date-1)
  horizon  : 1, 5 or 22
  forecast : average daily GK variance over the next `horizon` days
             (or volatility / annualised, see config.EXPORT_UNIT and ANNUALIZE)
"""
import numpy as np
import pandas as pd

import config


def residual_variance(log_pred: np.ndarray, log_true: np.ndarray) -> float:
    """s^2 of log forecast errors, used for the lognormal bias correction."""
    return float(np.var(log_true - log_pred))


def to_variance(log_pred, s2: float = 0.0) -> np.ndarray:
    """exp(E[log x]) underestimates E[x]; under log-normality
    E[x] = exp(E[log x] + s^2 / 2)."""
    adj = 0.5 * s2 if config.BIAS_CORRECTION else 0.0
    return np.exp(np.asarray(log_pred) + adj)


def to_export_units(variance) -> np.ndarray:
    v = np.asarray(variance) * (config.TRADING_DAYS if config.ANNUALIZE else 1)
    return np.sqrt(v) if config.EXPORT_UNIT == "volatility" else v


def contract_frame(dates, horizon: int, variance, model: str) -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.DatetimeIndex(dates).strftime("%Y-%m-%d"),
        "horizon": horizon,
        "forecast": to_export_units(variance),
        "model": model,
    })


def write_contract(frames: list[pd.DataFrame], path) -> pd.DataFrame:
    out = pd.concat(frames, ignore_index=True).sort_values(["horizon", "date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.10g")
    print(f"  wrote {len(out)} rows -> {path}")
    return out

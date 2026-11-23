"""Back-transform log forecasts and export them in the group contract (step 8).

Contract (root README, step 5): one CSV per phase with columns
    model_id, origin_date, horizon, prediction
  origin_date : forecast origin (made at 17:00 New York on that session)
  horizon     : 1, 5 or 22
  prediction  : expected average daily realized variance over the next
                `horizon` sessions, unannualized
Exactly one row per key of the phase's forecast_schedule.csv; the shared
validator in pipeline/csv_io.py enforces that.
"""
import numpy as np
import pandas as pd

import config
from data_loader import FORECAST, export


def residual_variance(log_pred, log_true) -> float:
    """s^2 of log forecast errors, used for the lognormal bias correction."""
    return float(np.var(np.asarray(log_true) - np.asarray(log_pred)))


def to_variance(log_pred, s2: float = 0.0) -> np.ndarray:
    """exp(E[log x]) underestimates E[x]; under log-normality
    E[x] = exp(E[log x] + s^2 / 2)."""
    adj = 0.5 * s2 if config.BIAS_CORRECTION else 0.0
    return np.exp(np.asarray(log_pred) + adj)


def contract_frame(dates, horizon: int, variance, model_id: str) -> pd.DataFrame:
    v = np.asarray(variance, dtype=float)
    return pd.DataFrame({
        "model_id": model_id,
        "origin_date": pd.DatetimeIndex(dates).strftime("%Y-%m-%d"),
        "horizon": horizon,
        # a failed forecast stays as a blank row, as the contract asks
        "prediction": np.where(np.isfinite(v) & (v > 0), v, np.nan),
    })


def write_contract(frames: list[pd.DataFrame], path, schedule: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat(frames, ignore_index=True)
    sched = schedule.assign(origin_date=schedule["origin_date"].dt.strftime("%Y-%m-%d"))
    export(out[list(FORECAST)], path, FORECAST, schedule=sched)
    print(f"  wrote {len(out)} rows ({int(out['prediction'].isna().sum())} blank) -> {path}")
    return out

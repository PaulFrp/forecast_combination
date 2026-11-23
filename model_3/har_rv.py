"""HAR-RV (Corsi 2009) direct forecasts with an expanding-window walk-forward.

Run from the repo root:
    python -m model_3.har_rv --phase validation
    python -m model_3.har_rv --phase test

For each scheduled (origin t, horizon h), in date order:
  1. Fit OLS  Y_{s,h} = b0 + b_d*RV_s + b_w*RV5_s + b_m*RV22_s  on every origin s with
     complete features and a matured label: pos[s] + h <= pos[t]  (expanding window, refit every origin).
  2. Predict from the features at t.
  3. If the prediction is not strictly positive, fall back to the persistence baseline (counted).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.csv_io import DAILY, FORECAST, HORIZONS, KEY, export, load

MODEL_ID = "har_rv"
DATA = Path("data/alpaca")
PHASES = ("train", "validation", "test")


def load_rv() -> pd.Series:
    """spy_rv on one continuous trading calendar (train + validation + test)."""
    daily = pd.concat([load(DATA / p / "daily_inputs.csv", DAILY) for p in PHASES],
                      ignore_index=True).set_index("date")
    if not (daily.index.is_unique and daily.index.is_monotonic_increasing):
        raise ValueError("calendar dates must be unique and increasing")
    return daily["spy_rv"]


def build_features(rv: pd.Series) -> pd.DataFrame:
    """HAR regressors at origin t; every window includes t."""
    return pd.DataFrame({
        "rv_d": rv,
        "rv_w": rv.rolling(5).mean(),
        "rv_m": rv.rolling(22).mean(),
    })


def ols_predict(X: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> float:
    """Fit OLS with an intercept and predict one row."""
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(beta[0] + x_new @ beta[1:])


def walk_forward(rv: pd.Series, schedule: pd.DataFrame) -> pd.DataFrame:
    """One forecast per schedule row, fitted only on labels matured by that origin."""
    F = build_features(rv).to_numpy()
    complete = np.isfinite(F).all(axis=1)
    pos = pd.Series(np.arange(len(rv)), index=rv.index)
    P = pos.to_numpy()
    targets = {h: rv.rolling(h).mean().shift(-h).to_numpy() for h in HORIZONS}
    baselines = {h: rv.rolling(h).mean().to_numpy() for h in HORIZONS}

    rows = []
    for origin, h in schedule[list(KEY)].itertuples(index=False, name=None):
        t = pos[origin]
        y = targets[h]
        train = complete & np.isfinite(y) & (P + h <= t)
        pred, fallback = np.nan, False
        if complete[t] and train.sum() > F.shape[1] + 1:
            pred = ols_predict(F[train], y[train], F[t])
            if not pred > 0:
                pred, fallback = baselines[h][t], True
        rows.append({"origin_date": origin, "horizon": h, "prediction": pred,
                     "fallback": fallback, "n_train": int(train.sum())})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phase", choices=("validation", "test"), required=True)
    args = parser.parse_args()

    schedule = load(DATA / args.phase / "forecast_schedule.csv", KEY)
    result = walk_forward(load_rv(), schedule)

    out = Path("forecasts") / args.phase / f"{MODEL_ID}.csv"
    export(result.assign(model_id=MODEL_ID)[list(FORECAST)], out, FORECAST, schedule=schedule)

    summary = result.groupby("horizon").agg(
        forecasts=("prediction", "size"),
        failed=("prediction", lambda s: int(s.isna().sum())),
        baseline_fallbacks=("fallback", "sum"),
        n_train_first=("n_train", "first"),
        n_train_last=("n_train", "last"),
    )
    print(f"Wrote {out}\n{summary.to_string()}")


if __name__ == "__main__":
    main()

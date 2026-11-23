"""Scaling and sliding windows (step 3).

Samples are identified by their *position* in the feature frame. A sample at
position i (origin t) sees the feature rows i-window+1 .. i, all of which only
use data through session t, and predicts target_h at row i.

Matured labels: a sample's target covers sessions t+1 .. t+h. At a fitting
cutoff c (an origin date, 17:00 New York) the label is usable only if t+h <= c,
i.e. pos[t] + h <= pos[c] in the trading calendar (group README, step 4).
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Scaler:
    """Plain standardisation; fitted on training rows only."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        std = x.std(axis=0)
        return cls(x.mean(axis=0), np.where(std > 0, std, 1.0))

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


def target_end_dates(index: pd.DatetimeIndex, h: int) -> pd.Series:
    """Last session covered by the h-session target on each row (NaT at the end)."""
    s = pd.Series(index, index=index)
    return s.shift(-h)


def select_samples(data: pd.DataFrame, h: int, window: int, start=None, end=None,
                   known_through=None, require_target=True, origins=None) -> np.ndarray:
    """Positions of usable samples.

    start / end      : keep samples whose origin t is in [start, end]
    known_through    : keep only samples whose whole target window ends on or
                       before this session (labels matured at that cutoff)
    require_target   : drop samples whose target is NaN
    origins          : keep only these dates (e.g. a phase's forecast schedule)
    """
    idx = data.index
    ok = np.arange(len(idx)) >= window - 1           # enough history for a window
    if start is not None:
        ok &= idx >= start
    if end is not None:
        ok &= idx <= end
    if known_through is not None:
        ends = target_end_dates(idx, h)
        ok &= (ends <= known_through).to_numpy()     # NaT compares False -> dropped
    if origins is not None:
        ok &= idx.isin(pd.DatetimeIndex(origins))
    if require_target:
        ok &= data[f"target_{h}"].notna().to_numpy()
    return np.flatnonzero(ok)


def make_windows(x: np.ndarray, positions: np.ndarray, window: int) -> np.ndarray:
    """(n_samples, window, n_features) array of past feature rows."""
    offsets = np.arange(-window + 1, 1)
    return x[positions[:, None] + offsets[None, :]]


@dataclass
class Prepared:
    """Everything a model needs for one (horizon, window, training cut-off)."""
    x_scaled: np.ndarray       # all feature rows, scaled with the training scaler
    x_scaler: Scaler
    y_mean: float              # target standardisation (training targets only)
    y_std: float

    def windows(self, positions, window):
        return make_windows(self.x_scaled, positions, window).astype(np.float32)


def prepare(data: pd.DataFrame, feature_cols: list[str], h: int,
            train_positions: np.ndarray) -> Prepared:
    """Fit feature + target scalers on the training samples only and scale all rows.

    The feature scaler is fitted on every feature row up to the last training
    sample (those rows are all known at training time)."""
    x = data[feature_cols].to_numpy(dtype=np.float64)
    last = train_positions.max()
    x_scaler = Scaler.fit(x[: last + 1])
    y = data[f"target_{h}"].to_numpy()[train_positions]
    return Prepared(x_scaler.transform(x), x_scaler, float(y.mean()), float(y.std()))


def split_positions(data: pd.DataFrame, h: int, window: int, splits: dict) -> dict[str, np.ndarray]:
    """Train / val sample positions for the initial fit (early stopping, tuning).

    train: training origins whose label matured by the last training session
           (= the nonblank labels of train/actuals.csv);
    val  : validation origins whose label matured by the last validation session
           (= the validation schedule)."""
    (_, tr_end), (va_start, va_end) = splits["train"], splits["val"]
    return {
        "train": select_samples(data, h, window, end=tr_end, known_through=tr_end),
        "val": select_samples(data, h, window, start=va_start, end=va_end, known_through=va_end),
    }

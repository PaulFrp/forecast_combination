"""HAR baseline (step 4): log-HAR regression estimated by OLS.

    target_h(t) = b0 + b_d * har_d(t) + b_w * har_w(t) + b_m * har_m(t) + e

Same samples, same split and same refit scheme as the LSTM, so it is the bar
the LSTM has to beat or match.
"""
import numpy as np
import pandas as pd

HAR_COLS = ["har_d", "har_w", "har_m"]


class HARModel:
    name = "HAR"

    def __init__(self, h: int):
        self.h = h
        self.coef = None

    def _design(self, data: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
        x = data[HAR_COLS].to_numpy()[positions]
        return np.column_stack([np.ones(len(x)), x])

    def fit(self, data: pd.DataFrame, train_pos: np.ndarray, val_pos=None) -> "HARModel":
        y = data[f"target_{self.h}"].to_numpy()[train_pos]
        self.coef, *_ = np.linalg.lstsq(self._design(data, train_pos), y, rcond=None)
        return self

    def predict(self, data: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
        """Forecast of log average variance."""
        return self._design(data, positions) @ self.coef

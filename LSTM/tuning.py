"""Small hyper-parameter search on the validation set only (step 6).

Models are trained on the training period with early stopping on validation;
configurations are ranked by validation MSE of the log forecast.
The test period is never touched here.
"""
import itertools
import random

import numpy as np
import pandas as pd

import config
from model import LSTMEnsemble
from sequences import split_positions


def candidate_configs(n: int = config.N_SEARCH, seed: int = 0) -> list[dict]:
    """Random subset of the grid, always including the default configuration."""
    keys = list(config.SEARCH_GRID)
    grid = [dict(zip(keys, vals)) for vals in itertools.product(*config.SEARCH_GRID.values())]
    random.Random(seed).shuffle(grid)
    configs = [dict(config.DEFAULT_PARAMS)] + [g for g in grid if g != config.DEFAULT_PARAMS]
    return configs[:n]


def tune_horizon(data: pd.DataFrame, feature_cols: list[str], splits: dict, h: int,
                 configs: list[dict]) -> pd.DataFrame:
    rows = []
    for k, params in enumerate(configs, 1):
        pos = split_positions(data, h, params["window"], splits)
        ens = LSTMEnsemble(h, feature_cols, params, seeds=config.SEARCH_SEEDS)
        ens.fit(data, pos["train"], pos["val"])
        y_val = data[f"target_{h}"].to_numpy()[pos["val"]]
        mse = float(np.mean((ens.predict(data, pos["val"]) - y_val) ** 2))
        rows.append({"horizon": h, **params, "val_mse_log": mse,
                     "best_epoch_mean": float(np.mean(ens.best_epochs))})
        print(f"  h={h:2d} [{k:2d}/{len(configs)}] {params} -> val MSE(log) {mse:.4f}")
    return pd.DataFrame(rows).sort_values("val_mse_log").reset_index(drop=True)


def tune(data, feature_cols, splits, horizons=config.HORIZONS) -> tuple[dict[int, dict], pd.DataFrame]:
    """Best parameters per horizon, and the full results table."""
    configs = candidate_configs()
    tables = [tune_horizon(data, feature_cols, splits, h, configs) for h in horizons]
    results = pd.concat(tables, ignore_index=True)
    keys = list(config.SEARCH_GRID)
    best = {}
    for h, table in zip(horizons, tables):
        row = table.iloc[0]
        best[h] = {k: (type(config.DEFAULT_PARAMS[k])(row[k])) for k in keys}
    return best, results

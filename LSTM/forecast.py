"""Out-of-sample forecasts for the test period (step 7).

Two schemes, for any model with fit / predict (HARModel or LSTMEnsemble):
  * static    : fit once on train+val, predict the whole test period;
  * expanding : at each refit date T (every month / quarter) fit on every sample
                whose target is fully observed before T, then predict until the
                next refit date. Like the econometric models, the model always
                sees the most recent data.
LSTM refits have no validation set, so they train for a fixed number of epochs
(the median early-stopping epoch found on validation).
"""
from typing import Callable

import numpy as np
import pandas as pd

import config
from sequences import select_samples


def refit_dates(test_start, test_end, freq: str) -> list[pd.Timestamp]:
    dates = pd.date_range(test_start, test_end, freq=freq)
    return sorted({pd.Timestamp(test_start), *dates})


def forecast_test(data: pd.DataFrame, h: int, window: int, splits: dict,
                  make_model: Callable[[], object], scheme: str = config.FORECAST_SCHEME,
                  freq: str = config.REFIT_FREQ, verbose: bool = True) -> pd.DataFrame:
    """DataFrame indexed by test date with columns `log_pred` and, for
    ensembles, `seed_<k>` (per-seed log forecasts)."""
    te_start, te_end = splits["test"]
    test_pos = select_samples(data, h, window, start=te_start, end=te_end, require_target=False)
    test_dates = data.index[test_pos]

    if scheme == "static":
        blocks = [(pd.Timestamp(te_start), test_pos)]
    elif scheme == "expanding":
        cuts = refit_dates(te_start, te_end, freq) + [pd.Timestamp(te_end) + pd.Timedelta(days=1)]
        blocks = [(a, test_pos[(test_dates >= a) & (test_dates < b)]) for a, b in zip(cuts[:-1], cuts[1:])]
        blocks = [(a, p) for a, p in blocks if len(p)]
    else:
        raise ValueError(f"unknown scheme {scheme!r}")

    frames = []
    for k, (cut, pos) in enumerate(blocks, 1):
        train_pos = select_samples(data, h, window, known_before=cut)
        model = make_model().fit(data, train_pos)
        out = pd.DataFrame(index=data.index[pos])
        if hasattr(model, "predict_per_seed"):
            per_seed = model.predict_per_seed(data, pos)
            for s, p in zip(model.seeds, per_seed):
                out[f"seed_{s}"] = p
            out["log_pred"] = per_seed.mean(axis=0)
        else:
            out["log_pred"] = model.predict(data, pos)
        frames.append(out)
        if verbose:
            print(f"    {getattr(model, 'name', '?'):>4} h={h:2d} refit {k:2d}/{len(blocks)} "
                  f"@ {cut.date()} ({len(train_pos)} train samples, {len(pos)} forecasts)")
    return pd.concat(frames)

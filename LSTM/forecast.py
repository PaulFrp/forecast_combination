"""Walk-forward forecasts for an evaluation phase (step 7).

Used for both the validation schedule (the forecasts the combination step learns
weights from) and the test schedule. Two schemes, for any model with
fit / predict (HARModel or LSTMEnsemble):
  * static    : fit once at the first origin of the phase, on every label
                matured by then, and predict the whole phase;
  * expanding : at each refit date T (every month / quarter) fit on every label
                matured by origin T, then predict until the next refit date.
LSTM refits have no validation set, so they train for a fixed number of epochs
(the median early-stopping epoch of the initial fit).
"""
from typing import Callable

import pandas as pd

import config
from sequences import select_samples


def refit_dates(origins: pd.DatetimeIndex, freq: str) -> list[pd.Timestamp]:
    """First scheduled origin on or after each calendar boundary."""
    bounds = pd.date_range(origins.min(), origins.max(), freq=freq)
    firsts = {origins[origins >= b].min() for b in bounds}
    return sorted({origins.min(), *firsts})


def forecast_phase(data: pd.DataFrame, h: int, window: int, origins,
                   make_model: Callable[[], object], scheme: str = config.FORECAST_SCHEME,
                   freq: str = config.REFIT_FREQ, verbose: bool = True) -> pd.DataFrame:
    """DataFrame indexed by origin date with columns `log_pred` and, for
    ensembles, `seed_<k>` (per-seed log forecasts)."""
    pos = select_samples(data, h, window, require_target=False, origins=origins)
    dates = data.index[pos]
    missing = pd.DatetimeIndex(origins).difference(dates)
    if len(missing):
        raise ValueError(f"h={h}: {len(missing)} scheduled origins lack features, e.g. {list(missing[:3])}")

    if scheme == "static":
        cuts = [dates.min()]
    elif scheme == "expanding":
        cuts = refit_dates(dates, freq)
    else:
        raise ValueError(f"unknown scheme {scheme!r}")
    ends = cuts[1:] + [dates.max() + pd.Timedelta(days=1)]

    frames = []
    for k, (cut, nxt) in enumerate(zip(cuts, ends), 1):
        block = pos[(dates >= cut) & (dates < nxt)]
        train_pos = select_samples(data, h, window, known_through=cut)
        model = make_model().fit(data, train_pos)
        out = pd.DataFrame(index=data.index[block])
        if hasattr(model, "predict_per_seed"):
            per_seed = model.predict_per_seed(data, block)
            for s, p in zip(model.seeds, per_seed):
                out[f"seed_{s}"] = p
            out["log_pred"] = per_seed.mean(axis=0)
        else:
            out["log_pred"] = model.predict(data, block)
        frames.append(out)
        if verbose:
            print(f"    {getattr(model, 'name', '?'):>4} h={h:2d} refit {k:2d}/{len(cuts)} "
                  f"@ {cut.date()} ({len(train_pos)} train samples, {len(block)} forecasts)")
    return pd.concat(frames)

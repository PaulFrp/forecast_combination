"""Load the group's Alpaca snapshot (SPY daily inputs, actuals, schedules)."""
import sys

import pandas as pd

import config

# The shared CSV reader/validator lives in <root>/pipeline.
if str(config.ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(config.ROOT_DIR))
from pipeline.csv_io import ACTUALS, DAILY, FORECAST, KEY, export, load  # noqa: E402,F401


def load_daily() -> pd.DataFrame:
    """train + validation + test daily_inputs.csv, one row per SPY session,
    indexed by date. Kept on the full calendar so lookbacks cross phase boundaries."""
    frames = [load(d / "daily_inputs.csv", DAILY) for d in config.PHASE_DIRS.values()]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    assert df.index.is_unique, "overlapping phases in daily_inputs.csv"
    return df


def load_split_dates() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """(first_session, last_session) of each phase."""
    out = {}
    for name, d in config.PHASE_DIRS.items():
        dates = pd.to_datetime(pd.read_csv(d / "daily_inputs.csv", usecols=["date"])["date"])
        out[name] = (dates.min(), dates.max())
    assert out["train"][1] < out["val"][0] <= out["val"][1] < out["test"][0], out
    return out


def load_schedule(phase: str) -> pd.DataFrame:
    """origin_date (Timestamp), horizon for a phase with a forecast schedule."""
    s = load(config.PHASE_DIRS[phase] / "forecast_schedule.csv", KEY)
    s["origin_date"] = pd.to_datetime(s["origin_date"])
    return s


def load_actuals(phase: str) -> pd.DataFrame:
    """origin_date, horizon, actual, baseline (blank -> NaN)."""
    a = load(config.PHASE_DIRS[phase] / "actuals.csv", ACTUALS)
    a["origin_date"] = pd.to_datetime(a["origin_date"])
    for c in ("actual", "baseline"):
        a[c] = pd.to_numeric(a[c].replace("", float("nan")))
    return a


def load_all() -> tuple[pd.DataFrame, dict]:
    """Daily inputs on the full calendar, and the phase date ranges."""
    return load_daily(), load_split_dates()

"""Central configuration for the LSTM volatility-forecasting pipeline.

Everything that is a *choice* (horizons, units, hyper-parameters, search grid,
refit frequency, ...) lives here so the other modules stay purely mechanical.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
LSTM_DIR = Path(__file__).resolve().parent
ROOT_DIR = LSTM_DIR.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = LSTM_DIR / "outputs"

ALPACA_DIR = DATA_DIR / "alpaca"
# The group's Alpaca snapshot: one folder per phase, each with daily_inputs.csv
# and actuals.csv (validation / test also have forecast_schedule.csv).
PHASES = {"train": "train", "val": "validation", "test": "test"}
PHASE_DIRS = {k: ALPACA_DIR / v for k, v in PHASES.items()}

# Submission files for the group (only written in full mode).
FORECASTS_DIR = ROOT_DIR / "forecasts"

# --------------------------------------------------------------------------- #
# Target / forecast contract
# --------------------------------------------------------------------------- #
HORIZONS = [1, 5, 22]
MODEL_NAME = "LSTM"
MODEL_ID = "lstm"           # model_id column of the submission CSVs

# Target (group contract): average daily realized variance `spy_rv` over the
# h sessions *after* the origin, in unannualized variance units.
# Floor applied to RV before taking logs.
RV_FLOOR = 1e-10

# --------------------------------------------------------------------------- #
# Features (information through the origin session, see features.py)
# --------------------------------------------------------------------------- #
USE_VIX = True
# Sessions by which vix_close is lagged. The Cboe history has no as-of
# timestamps, so (like the Timestral runs) we only use the previous session's close.
VIX_LAG = 1
HAR_WINDOWS = {"har_d": 1, "har_w": 5, "har_m": 22}

# --------------------------------------------------------------------------- #
# LSTM defaults (= the "tiny untuned" model used in quick mode)
# --------------------------------------------------------------------------- #
DEFAULT_PARAMS = {
    "window": 22,        # sequence length (past days fed to the LSTM)
    "units": 16,         # LSTM hidden size
    "dropout": 0.2,
    "lr": 1e-3,
}
BATCH_SIZE = 64
MAX_EPOCHS = 200
PATIENCE = 15              # early-stopping patience (epochs)
WEIGHT_DECAY = 0.0

SEEDS = [0, 1, 2, 3, 4]    # ensemble of 5 seeds, predictions averaged
QUICK_SEEDS = [0]          # quick mode: one seed, just to get forecasts out

# --------------------------------------------------------------------------- #
# Hyper-parameter search (validation set only - never the test set)
# --------------------------------------------------------------------------- #
SEARCH_GRID = {
    "window": [22, 44, 66],
    "units": [16, 32],
    "dropout": [0.1, 0.3],
    "lr": [1e-3, 3e-3],
}
N_SEARCH = 10              # random subset of the grid (keep the search small)
SEARCH_SEEDS = [0, 1]      # seeds averaged per configuration while tuning

# --------------------------------------------------------------------------- #
# Test-period forecasting
# --------------------------------------------------------------------------- #
#   "static"    -> train once on train+val, predict the whole test period
#   "expanding" -> refit on all data available at each refit date
FORECAST_SCHEME = "expanding"
REFIT_FREQ = "QS"          # pandas offset: "MS" = monthly, "QS" = quarterly

# Bias correction for exp(E[log x]) < E[x]:  multiply by exp(0.5 * s^2),
# with s^2 the residual variance of log forecasts on the validation set.
BIAS_CORRECTION = True

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
# "Crisis" days in the test set: ex-ante regime defined by the lagged monthly
# log-variance (har_m) being above this quantile of its *training* distribution.
CRISIS_QUANTILE = 0.80

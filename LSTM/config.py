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

FULL_DATA = DATA_DIR / "sp500_20y_daily.csv"
# The group's split files: only their *date ranges* are used, so that our
# train / validation / test periods match everyone else's exactly.
SPLIT_FILES = {
    "train": DATA_DIR / "sp500_train.csv",
    "val": DATA_DIR / "sp500_val.csv",
    "test": DATA_DIR / "sp500_test.csv",
}
VIX_CACHE = DATA_DIR / "vix_daily.csv"   # downloaded once, then reused
VIX_TICKER = "^VIX"

# --------------------------------------------------------------------------- #
# Target / forecast contract
# --------------------------------------------------------------------------- #
HORIZONS = [1, 5, 22]
MODEL_NAME = "LSTM"

# Units of the exported forecast. Change these if the group agrees otherwise.
#   "variance"   -> Garman-Klass daily variance (average over the next h days)
#   "volatility" -> sqrt of the above
EXPORT_UNIT = "variance"
ANNUALIZE = False          # if True: variance * 252  (volatility * sqrt(252))
TRADING_DAYS = 252

# Floor applied to GK variance before taking logs (a few days have H == L).
GK_FLOOR = 1e-8

# --------------------------------------------------------------------------- #
# Features (all lagged by one day inside features.py)
# --------------------------------------------------------------------------- #
USE_VIX = True             # silently disabled if VIX cannot be downloaded
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

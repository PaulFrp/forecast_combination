# LSTM volatility forecasts

Forecasts the expected average daily realized variance of SPY (`spy_rv`, 5-minute RV) over the next
1, 5 and 22 sessions with a small LSTM, using the group's Alpaca snapshot in `data/alpaca/`.
It is benchmarked against the group's HAR-RV model (`model_3/`, read from `forecasts/<phase>/har_rv.csv`)
and against the persistence baseline. Those files must exist before running; regenerate them with
`python -m model_3.har_rv --phase validation` and `--phase test`.

```bash
python LSTM/pipeline.py --mode quick   # ~30 s: tiny untuned LSTM, 1 seed, static fits
python LSTM/pipeline.py --mode full    # tuning on validation, 5 seeds, quarterly expanding refits
```
Options: `--scheme static|expanding`, `--no-tune`, `--horizons 1 5`, `--no-submit`. All settings are in `config.py`.
A full run over all horizons also writes the group submissions `forecasts/validation/lstm.csv` and
`forecasts/test/lstm.csv` (`model_id=lstm`).

## Files
| file | step |
|---|---|
| `config.py` | every choice: horizons, hyper-parameters, grid, refit frequency, VIX lag |
| `data_loader.py` | train/validation/test `daily_inputs.csv` on one calendar, schedules, actuals, model_3 HAR-RV forecasts |
| `features.py` | 1–2: targets and features |
| `sequences.py` | 3: train-only scaling, sliding windows, matured-label filtering |
| `model.py` | 5: 1-layer LSTM, Adam, MSE, early stopping, seed ensemble |
| `tuning.py` | 6: small random search, validation set only |
| `forecast.py` | 7: walk-forward (static or expanding) forecasts over a phase's schedule |
| `export.py` | 8: back to variance, lognormal bias correction, contract CSV (shared validator) |
| `evaluate.py` | 9: RMSE / QLIKE vs model_3 HAR-RV and persistence, Diebold–Mariano, calm vs crisis, seed stability, permutation importance |
| `plots.py` | figures for the slides |
| `pipeline.py` | runs everything |

## Conventions
* **Data**: `data/alpaca/{train,validation,test}/daily_inputs.csv` concatenated on the full trading calendar,
  so lookbacks cross phase boundaries. Train 2016-01-04 → 2020-12-31, validation 2021 → 2022, test
  2023-01-03 → 2026-11-13.
* **Timing**: row `t` is a forecast origin (17:00 New York on session `t`). Features use data through `t`.
  `target_h(t) = log(mean(spy_rv[t+1 .. t+h]))`, which is the log of `actual` in `actuals.csv`. The pipeline
  checks this match exactly.
* **Features**: `har_d/har_w/har_m` (log of mean `spy_rv` over the last 1/5/22 sessions), `ret_1`,
  `neg_ret_1` (leverage effect), `ret_5`, `log_volume` (relative to its 66-session mean) and `log_vix`.
  `log_vix` uses the *previous* session's `vix_close` (`VIX_LAG = 1`), because the Cboe history has no as-of
  timestamps. Warm-up: the first 66 sessions (through 2016-04-06) are dropped.
* **Fitting**: only matured labels are used: at cutoff `c`, a label at origin `t` is usable if `t+h ≤ c`.
  The initial fit (train, with early stopping on validation) chooses hyper-parameters and the epoch count.
  Validation and test forecasts are then produced walk-forward over each phase's `forecast_schedule.csv`.
  In full mode, the model is refitted at the first origin of every quarter on all labels matured by then,
  for the median early-stopping epoch count.
* **Back to variance**: the model predicts log variance, and `prediction = exp(log_pred + s²/2)`.
  `s²` is the variance of the walk-forward validation log errors, per horizon.
* **Export**: `model_id,origin_date,horizon,prediction`, one row per scheduled key, unannualized variance,
  checked with `pipeline/csv_io.py`. A failed forecast is a blank prediction.

## Outputs (`outputs/<mode>/`)
`forecasts/{validation,test}/lstm.csv`, `validation.csv` and
`test_metrics.csv` (RMSE, QLIKE, ratios and DM p-values vs HAR-RV and persistence, by regime),
`seed_stability.csv`, `feature_importance.csv` (full), `tuning_results.csv` (full), `params.json`,
`test_detail.csv`, `*.png`.

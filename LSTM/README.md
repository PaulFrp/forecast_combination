# LSTM volatility forecasts

Forecasts the log Garman–Klass variance of the S&P 500 at horizons 1, 5 and 22 days
with a small LSTM, benchmarked against a HAR regression on the same split.

```bash
python LSTM/pipeline.py --mode quick   # ~30 s: tiny untuned LSTM, 1 seed, static fit
python LSTM/pipeline.py --mode full    # tuning on validation, 5 seeds, quarterly expanding refits
```
Options: `--scheme static|expanding`, `--no-tune`, `--horizons 1 5`. All settings are in `config.py`.

## Files
| file | step |
|---|---|
| `config.py` | every choice: horizons, units, hyper-parameters, grid, refit frequency |
| `data_loader.py` | prices, Garman–Klass variance, VIX download (cached in `data/vix_daily.csv`), group split dates |
| `features.py` | 1–2: targets and features |
| `sequences.py` | 3: train-only scaling, sliding windows, chronological split with purging |
| `har.py` | 4: HAR baseline (OLS on log HAR components) |
| `model.py` | 5: 1-layer LSTM, Adam, MSE, early stopping, seed ensemble |
| `tuning.py` | 6: small random search, validation set only |
| `forecast.py` | 7: static or expanding-window test forecasts |
| `export.py` | 8: back to variance, lognormal bias correction, contract CSV |
| `evaluate.py` | 9: QLIKE / MSE vs HAR, Diebold–Mariano, calm vs crisis, seed stability, permutation importance |
| `plots.py` | figures for the slides |
| `pipeline.py` | runs everything |

## Conventions
* **Split**: the date ranges of `data/sp500_{train,val,test}.csv`, so the test period is the group's
  (2023-11-13 → 2026-11-16). Train/val samples whose h-day target would run into the next period are dropped.
* **Timing**: row `t` = forecast made at the close of `t-1`. Every feature uses data up to `t-1`.
  `target_h(t) = log(mean(gk_var[t .. t+h-1]))`.
* **Features**: `har_d/har_w/har_m` (log of mean GK variance over the last 1/5/22 days), `ret_1`,
  `neg_ret_1` (leverage effect), `ret_5`, `log_volume` (relative to its 66-day mean, because raw volume
  trends over 20 years) and `log_vix`.
* **Export** (`outputs/<mode>/forecasts_lstm.csv`): `date, horizon, forecast, model`. `date` is the first
  day of the forecast window, and `forecast` is the average daily GK variance over the next `horizon` days
  (not annualized). Change `EXPORT_UNIT` / `ANNUALIZE` in `config.py` if the group uses other units.
  Bias correction: `exp(log_pred + s²/2)`, where `s²` is the residual variance of the log forecasts on
  validation.

## Outputs (`outputs/<mode>/`)
`forecasts_lstm.csv`, `forecasts_har_baseline.csv`, `validation.csv`, `test_metrics.csv`, `seed_stability.csv`,
`feature_importance.csv` (full), `tuning_results.csv` (full), `params.json`, `test_detail.csv`, `*.png`.

# HAR-RV — `model_id = har_rv`

Econometrics family: Heterogeneous Autoregressive model of realized variance (Corsi, 2009), direct multi-horizon form, estimated by OLS in variance levels.

## Run

From the repository root, with the project `.venv` active:

```sh
python -m model_3.har_rv --phase validation   # -> forecasts/validation/har_rv.csv
python -m model_3.har_rv --phase test         # -> forecasts/test/har_rv.csv
```

Each run prints per-horizon counts of forecasts, failed forecasts, baseline fallbacks, and training-sample size. The script is deterministic (≈0.5 s) and needs only the root `requirements.txt`. `model_3/requirements.txt` adds `statsmodels`, used only by `explore.ipynb` for HAC t-statistics.

Score with the shared report:

```sh
python -m pipeline.report --schedule data/alpaca/<phase>/forecast_schedule.csv \
  --actuals data/alpaca/<phase>/actuals.csv \
  --forecasts forecasts/<phase>/har_rv.csv --selected-id har_rv \
  --output model_3/results/<phase>
```

Running `explore.ipynb` top to bottom regenerates everything in `model_3/results/` (the same report, plus the log-scale and in-sample plots).

## Model

For each horizon h ∈ {1, 5, 22}, a separate regression (same regressors, different target):

$$
Y_{t,h} = \beta_0 + \beta_d\,RV_t + \beta_w\,RV^{(5)}_t + \beta_m\,RV^{(22)}_t + \varepsilon_{t,h},
\qquad Y_{t,h} = \tfrac{1}{h}\textstyle\sum_{j=1}^{h} RV_{t+j}
$$

| Input | Definition | Lookback |
|---|---|---|
| `rv_d` | `spy_rv` at origin t | 1 session |
| `rv_w` | mean of `spy_rv` over t−4 … t | 5 sessions |
| `rv_m` | mean of `spy_rv` over t−21 … t | 22 sessions |

- Only `spy_rv` is used. No VIX, no OHLCV, no `actual`/`baseline` columns.
- Features are built on one continuous calendar (train + validation + test `daily_inputs.csv`), so lookbacks carry across split boundaries. A feature row is used only if all three windows are complete; `spy_rv` has no gaps in the Alpaca data, so this only drops the first 21 sessions of 2016.
- Predictions are in unannualized daily variance units. The model is fit in levels, so **no log-variance conversion** is applied.

## Data snapshot and cutoff

- `data/alpaca/` as of commit `665a831` (2,733 sessions, 2016-01-04 to 2026-11-13). `data/yahoo/` is not used.
- Information cutoff: 17:00 New York on the origin date. All features at t use `spy_rv` through t only.
- The two estimated training variances (2016-02-02, 2019-08-12) enter training labels and features. Their availability dates are long before the first forecast origin (2021-01-04), so they create no look-ahead in validation or test.

## Fit and update policy

- **Window:** expanding. Training starts at the first complete feature row (2016-02-03).
- **Refit:** at every forecast origin, separately per horizon, in date order.
- **Matured labels only:** at origin t, a training row s is used only if `pos[s] + h <= pos[t]` (positions in the full trading calendar) and its target is complete. Labels from validation are added as they mature during the test run.
- **No tuning.** The configuration was fixed before validation: no window search, no log-HAR, no variant selection. Validation was used only to check the model. The script was not changed between the validation and test runs.
- Training-set size at the first → last origin (h = 1): validation 1,238 → 1,718; test 1,741 → 2,689.

Leakage check: for 25 random validation origins, re-running the walk-forward on `spy_rv` truncated at the origin gives identical predictions (max abs difference 0).

## Non-positive and failed forecasts

- **Rule:** if the OLS prediction is not strictly positive, the persistence baseline (mean of the last h `spy_rv` values through t) is submitted instead.
- **Fallbacks used:** 0 in validation, 0 in test.
- **Failed forecasts (blank prediction):** 0 in validation, 0 in test. A blank would only occur if the origin's features were incomplete or the training sample had fewer than 5 rows; neither happens.

## Results (RMSE, unannualized variance)

| Phase | Horizon | har_rv | Baseline | Change | Scored / scheduled |
|---|---:|---:|---:|---:|---:|
| Validation | 1 | 7.30e-5 | 8.31e-5 | −12.2% | 481 / 481 |
| Validation | 5 | 6.06e-5 | 6.59e-5 | −8.0% | 481 / 481 |
| Validation | 22 | 5.37e-5 | 5.61e-5 | −4.3% | 481 / 481 |
| Test | 1 | 1.50e-4 | 1.68e-4 | −10.3% | 949 / 949 |
| Test | 5 | 1.22e-4 | 1.38e-4 | −11.9% | 949 / 949 |
| Test | 22 | 8.23e-5 | 8.81e-5 | −6.6% | 949 / 949 |

## Known limitations

- **Outlier sensitivity.** OLS in levels is dominated by extreme days. In the train fit, March 2020 turns the monthly coefficient negative (≈ −0.10) and raises the intercept. Pre-COVID estimates show the usual HAR pattern (see `explore.ipynb`, Step 3). The inflated intercept makes 22-session forecasts run high in calm periods.
- **April 2025.** The tariff-shock spike dominates test RMSE for every method. At h = 22, HAR reacts faster than the baseline but overshoots for a few days after the spike.
- Log-HAR and rolling windows were deliberately not tried. They are natural extensions if outlier robustness matters more than simplicity.

## Files

| File | Purpose |
|---|---|
| `har_rv.py` | Walk-forward forecasting and CSV export (the submission code) |
| `explore.ipynb` | Steps 1–3: target/baseline checks against `actuals.csv`, export plumbing check, in-sample fit with HAC t-stats and pre-COVID sensitivity |
| `requirements.txt` | `statsmodels` for the notebook only |
| `results/<phase>/report.md`, `scores.csv`, `coverage.csv` | Shared-report RMSE table and coverage for validation and test |
| `results/<phase>/forecasts.png` | Shared-report forecast plot (linear scale) |
| `results/<phase>/forecasts_log.png` | Same plot on a log scale; calm periods readable |
| `results/train_insample_fit.png` | Step 3 in-sample fit on train (log scale) |
